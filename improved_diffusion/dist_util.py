"""
Helpers for distributed training.
"""

import io
import os
import socket
from typing import Optional

import blobfile as bf
import torch as th
import torch.distributed as dist

# Try to import MPI, but make it optional.
try:
    from mpi4py import MPI  # type: ignore
    HAVE_MPI = True
except Exception:
    HAVE_MPI = False

# If you use torchrun, prefer LOCAL_RANK mapping; otherwise fall back.
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "8"))
SETUP_RETRY_COUNT = 3

# Global cached device decided by setup_dist()
_DEV: th.device = th.device("cpu")


# ---------------------------
# Public API
# ---------------------------

def setup_dist() -> None:
    """
    Setup single-process or distributed process group.

    Modes:
      1) OPENAI_USE_DDP=0  -> single process (no dist.init_process_group)
      2) torchrun env      -> use env RANK/WORLD_SIZE/LOCAL_RANK
      3) MPI fallback      -> use MPI to bcast master addr/port & set envs
    """
    global _DEV

    # If we've already set up, just return.
    if dist.is_available() and dist.is_initialized():
        _DEV = _infer_device_after_init()
        return

    use_ddp = int(os.environ.get("OPENAI_USE_DDP", "1")) == 1

    # -------------------
    # Decide if we should init DDP
    # -------------------
    env_world_size = int(os.environ.get("WORLD_SIZE", "0")) or None
    mpi_world_size = _mpi_world_size() if HAVE_MPI else 1

    should_init_ddp = use_ddp and (
        (env_world_size is not None and env_world_size > 1) or
        (HAVE_MPI and mpi_world_size > 1)
    )

    if not should_init_ddp:
        # Single process path: bind to GPU:0 if available.
        if th.cuda.is_available():
            th.cuda.set_device(0)
            _DEV = th.device("cuda", 0)
        else:
            _DEV = th.device("cpu")
        return

    # -------------------
    # Build env for init (torchrun or MPI)
    # -------------------
    backend = "nccl" if th.cuda.is_available() else "gloo"

    if _has_torchrun_env():
        # torchrun should already provide MASTER_ADDR/PORT/RANK/WORLD_SIZE/LOCAL_RANK
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Important: set device BEFORE init to silence the NCCL warning.
        if th.cuda.is_available():
            th.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        _DEV = _infer_device_after_init(default_local_rank=local_rank)
        return

    # Fallback to MPI-coordinated env if not using torchrun
    if HAVE_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # Decide master addr/port
        if backend == "gloo":
            hostname = "127.0.0.1"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())

        master_addr = comm.bcast(hostname, root=0)
        master_port = comm.bcast(_find_free_port() if rank == 0 else None, root=0)

        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Map rank -> device
        local_rank = rank % max(1, _num_visible_gpus())
        if th.cuda.is_available():
            th.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend, init_method="env://")
        _DEV = _infer_device_after_init(default_local_rank=local_rank)
        return

    # Should not reach here (we already checked should_init_ddp)
    # But as a safety, do single process.
    if th.cuda.is_available():
        th.cuda.set_device(0)
        _DEV = th.device("cuda", 0)
    else:
        _DEV = th.device("cpu")


def dev() -> th.device:
    """
    Return the device selected by setup_dist().
    """
    return _DEV


def load_state_dict(path: str, **kwargs):
    """
    Load a PyTorch checkpoint with minimal I/O in DDP:
      - If dist initialized: rank 0 reads, then broadcast bytes to all ranks.
      - Else: read locally.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
        else:
            data = None
        obj_list = [data]
        dist.broadcast_object_list(obj_list, src=0)
        data = obj_list[0]
        assert data is not None, "Broadcasted checkpoint bytes are None."
        return th.load(io.BytesIO(data), **kwargs)
    else:
        with bf.BlobFile(path, "rb") as f:
            return th.load(f, **kwargs)


def sync_params(params) -> None:
    """
    Broadcast parameters from rank 0 to all ranks if DDP is initialized.
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    for p in params:
        dist.broadcast(p, src=0)


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------
# Helpers
# ---------------------------

def _has_torchrun_env() -> bool:
    # torchrun typically sets these
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _num_visible_gpus() -> int:
    if not th.cuda.is_available():
        return 0
    # Respect CUDA_VISIBLE_DEVICES
    return th.cuda.device_count()


def _infer_device_after_init(default_local_rank: Optional[int] = None) -> th.device:
    """
    Infer device after dist.init_process_group(). If CUDA is available, prefer
    LOCAL_RANK; otherwise fall back to given default or 0.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", str(default_local_rank or 0)))
        return th.device("cuda", local_rank)
    return th.device("cpu")


def _mpi_world_size() -> int:
    if not HAVE_MPI:
        return 1
    try:
        return MPI.COMM_WORLD.Get_size()
    except Exception:
        return 1


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
