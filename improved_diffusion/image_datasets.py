# improved_diffusion/image_datasets.py
# -*- coding: utf-8 -*-

import os
import glob
from typing import Optional, Iterator

import numpy as np
import tifffile as tiff
from torch.utils.data import DataLoader, Dataset

# --- Optional: be tolerant if mpi4py is not present ---
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except Exception:
    _MPI_AVAILABLE = False


# =====================================================
# Normalization range (edit if your data range differs)
# =====================================================
CLIP_MIN = -50.0
CLIP_MAX =  50.0


def _to_n11(x: np.ndarray,
            clip_min: float = CLIP_MIN,
            clip_max: float = CLIP_MAX) -> np.ndarray:
    """
    Clamp to [clip_min, clip_max] and linearly map to [-1, 1].
    Works for both GT and cond so the model sees a consistent range.
    """
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, clip_min, clip_max)
    x = (x - clip_min) / (clip_max - clip_min) * 2.0 - 1.0
    return x


def denorm_from_n11(x: np.ndarray,
                    clip_min: float = CLIP_MIN,
                    clip_max: float = CLIP_MAX) -> np.ndarray:
    """
    Map from [-1, 1] back to the physical range [clip_min, clip_max].
    Useful for visualization with colorbars in [-50, 50].
    """
    x = x.astype(np.float32, copy=False)
    return 0.5 * (x + 1.0) * (clip_max - clip_min) + clip_min


def load_data(
    *,
    data_dir: str,
    batch_size: int,
    image_size: Optional[int] = None,   # not used unless you add resizing/tiling
    class_cond: bool = False,           # kept for API compat; unused
    deterministic: bool = False,        # True -> fixed order (e.g., validation)
    infinite: bool = True,              # True -> infinite iterator (training)
    num_workers: int = 4,               # speed up IO a bit
):
    """
    InSAR phase-unwrapping dataset loader (cond-only).

    Directory layout (filenames must match):
        <data_dir>/unwrapped/*.tif
        <data_dir>/cond/*.tif

    Returns:
        - if infinite=True: an iterator yielding batches forever (for training)
        - if infinite=False: (loader, len(dataset)) for single-pass validation
    """
    if not data_dir:
        raise ValueError("Please specify data_dir")

    # Collect GT (unwrapped) files
    unwrapped_files = sorted(glob.glob(os.path.join(data_dir, "unwrapped", "*.tif"))) + \
                      sorted(glob.glob(os.path.join(data_dir, "unwrapped", "*.tiff")))
    if len(unwrapped_files) == 0:
        raise ValueError(f"No unwrapped .tif/.tiff files found in {data_dir}/unwrapped")

    # Shard by MPI rank if available
    if _MPI_AVAILABLE:
        rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
    else:
        rank, world_size = 0, 1

    # Take every world_size-th file starting from this rank
    shard_files = unwrapped_files[rank::world_size]

    dataset = InSARPhaseDataset(unwrapped_list=shard_files)

    # For validation, we typically want drop_last=False and deterministic ordering.
    drop_last = False if deterministic or not infinite else True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=(num_workers > 0 and infinite),
    )

    if infinite:
        def _infinite_loader() -> Iterator[dict]:
            while True:
                for batch in loader:
                    yield batch
        return _infinite_loader()
    else:
        return loader, len(dataset)


class InSARPhaseDataset(Dataset):
    """
    For each:   <data_dir>/unwrapped/<name>.tif
    Expect:     <data_dir>/cond/<name>.tif
    Returns:
        {
            "image": [C, H, W] float32 in [-1, 1]  (GT unwrapped phase),
            "cond":  [C, H, W] float32 in [-1, 1]  (conditioning image),
            "name":  <str> base filename without extension
        }
    """
    def __init__(self, unwrapped_list):
        super().__init__()
        self.unwrapped_list = unwrapped_list

    def __len__(self) -> int:
        return len(self.unwrapped_list)

    def __getitem__(self, idx: int):
        unwrapped_path = self.unwrapped_list[idx]
        name = os.path.splitext(os.path.basename(unwrapped_path))[0]

        # Build cond path robustly based on folder names
        root_dir = os.path.dirname(os.path.dirname(unwrapped_path))  # -> data_dir
        cond_path = os.path.join(root_dir, "cond", os.path.basename(unwrapped_path))
        if not os.path.exists(cond_path):
            raise FileNotFoundError(f"Missing cond for {name}: {cond_path}")

        # Read raw float32 arrays (physical units, e.g., ~[-50, 50])
        unwrapped = tiff.imread(unwrapped_path).astype(np.float32)
        cond      = tiff.imread(cond_path).astype(np.float32)

        for path, array in ((unwrapped_path, unwrapped), (cond_path, cond)):
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite phase values in {path}")

        # Normalize both to [-1, 1] so the model sees a stable range
        unwrapped = _to_n11(unwrapped)  # -> [-1, 1]
        cond      = _to_n11(cond)       # -> [-1, 1]

        # Ensure [C, H, W]
        if unwrapped.ndim == 2:
            unwrapped = np.expand_dims(unwrapped, axis=0)
        elif unwrapped.ndim == 3 and unwrapped.shape[0] != 1:
            unwrapped = unwrapped[:1, ...]
        if cond.ndim == 2:
            cond = np.expand_dims(cond, axis=0)
        elif cond.ndim == 3 and cond.shape[0] != 1:
            cond = cond[:1, ...]

        return {"image": unwrapped, "cond": cond, "name": name}
