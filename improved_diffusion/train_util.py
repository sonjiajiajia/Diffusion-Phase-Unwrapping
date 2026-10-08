import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


def compute_nrmse_batch(gt: th.Tensor, pred: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    """
    Per-sample NRMSE for a batch.
    gt, pred: [B, 1, H, W]
    Returns: [B] NRMSE values normalized by (max(gt)-min(gt)) per sample.
    """
    assert gt.shape == pred.shape and gt.dim() == 4 and gt.shape[1] == 1
    B = gt.shape[0]
    g = gt.view(B, -1)
    p = pred.view(B, -1)
    rmse = th.sqrt(th.mean((g - p) ** 2, dim=1))  # [B]
    gmax, _ = th.max(g, dim=1)
    gmin, _ = th.min(g, dim=1)
    denom = (gmax - gmin).clamp_min(eps)
    return rmse / denom


class TrainLoop:
    """
    Training loop with:
      - microbatching
      - optional FP16
      - EMA weights
      - periodic validation with EMA & visualizations
      - robust logging (skip large tensors)
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_grad_norm=1.0,
        progress_interval: int = 100,
        val_max_batches=0,
        val_max_samples=0,

        # Validation controls
        val_loader=None,             # finite DataLoader (single pass)
        val_len=0,                   # optional info only
        val_every=2000,              # validate every N steps (skips step 0)
        val_num_vis=8,               # save up to K visualizations per validation
        val_save_dir="./val_vis",
        val_use_ddim=True,           # use DDIM sampling for faster/cleaner val
        val_timestep_respacing=None,
        val_diffusion=None,
    ):
        self.progress_interval = progress_interval
        self.val_max_batches = int(val_max_batches or 0)
        self.val_max_samples = int(val_max_samples or 0)

        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.max_grad_norm = float(max_grad_norm)
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and non-negative")

        # Validation state
        self.val_loader = val_loader
        self.val_len = val_len
        self.val_every = val_every
        self.val_num_vis = val_num_vis
        self.val_save_dir = val_save_dir
        self.val_use_ddim = val_use_ddim
        self.val_timestep_respacing = val_timestep_respacing
        self.val_diffusion = val_diffusion if val_diffusion is not None else diffusion

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Load checkpoint & sync params
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        # Optimizer
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # EMA buffers
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

        # DDP (respect OPENAI_USE_DDP=0 to force single-process)
        if th.cuda.is_available() and int(os.environ.get("OPENAI_USE_DDP", "1")):
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    # ---------- Setup / Checkpoint I/O ----------

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint and dist_util.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self._state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06d}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    # ---------- Main loop ----------

    def run_loop(self):
        while (not self.lr_anneal_steps) or (self.step + self.resume_step < self.lr_anneal_steps):
            batch = next(self.data)
            self.run_step(batch)

            # Logging
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # Checkpoint (skip step 0)
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            # Validation (skip step 0)
            if (
                self.val_loader is not None
                and self.val_every > 0
                and self.step > 0
                and self.step % self.val_every == 0
            ):
                self.validate()

            self.step += 1

        # Final save if we ended off-interval (and we actually trained)
        if self.step > 0 and (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    # ---------- Forward/Backward ----------

    def forward_backward(self, batch):
        # Reset gradients
        zero_grad(self.model_params)

        # Ground truth (x) and condition (cond)
        x = batch["image"]               # [B, 1, H, W]
        cond = {"cond": batch["cond"]}   # [B, Cc, H, W] (Cc=1 for your cond)
        if "valid_mask" in batch:
            cond["loss_mask"] = batch["valid_mask"]

        # Microbatch for memory efficiency
        for i in range(0, x.shape[0], self.microbatch):
            micro_x = x[i : i + self.microbatch].to(dist_util.dev(), non_blocking=True)
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev(), non_blocking=True)
                          for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= x.shape[0]

            # Sample diffusion timesteps
            t, weights = self.schedule_sampler.sample(micro_x.shape[0], dist_util.dev())

            # Loss closure
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_x,
                t,
                model_kwargs=micro_cond,  # cond-only
            )

            # DDP sync only on last microbatch to save comms
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # Update sampler if loss-aware
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            # Scalar loss
            loss = (losses["loss"] * weights).sum() / x.shape[0]
            if not th.isfinite(loss).all():
                names = batch.get("name", [])
                raise FloatingPointError(
                    f"Non-finite loss at step {self.step + self.resume_step}; "
                    f"samples={names[i:i + self.microbatch]}. "
                    "Training stopped before the optimizer update. "
                    "Check the data and restart with --use_fp16 False."
                )

            # "Safe" logging payload: keep only small scalars/vectors; skip big tensors.
            safe_losses = {}
            for k, v in losses.items():
                if k in ("loss", "mse", "vb"):
                    safe_losses[k] = (v * weights).detach()
                elif isinstance(v, th.Tensor) and v.ndim <= 1:
                    safe_losses[k] = v.detach()
                # else: skip large tensors (pred_xstart, model_output, x_t, eps, ...)

            log_loss_dict(self.diffusion, t, safe_losses)

            # Backprop (with optional FP16 scale)
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    # ---------- Validation every N steps ----------

    import math
    import os
    import torch as th
    import torch.distributed as dist
    from .fp16_util import master_params_to_model_params

    def validate(self):
        """
        Finite validation with caps:
          - Swap in EMA (if any), eval() model.
          - Iterate only up to val_max_batches or val_max_samples (if set);
            otherwise fall back to floor(val_len / bs) with a safety cap.
          - Compute mean NRMSE; save up to val_num_vis *combined* images (cond|pred|gt in one figure).
          - Optional all_reduce across ranks.
          - Restore training weights and train() afterward.
        """
        # Save the current training weights.
        logger.log("Validating...")
        orig_master = [p.detach().clone() for p in self.master_params]
        try:
            # Switch to EMA weights if available.
            if getattr(self, "ema_params", None):
                self.model.load_state_dict(self._master_params_to_state_dict(self.ema_params[0]))

            self.model.eval()
            model_for_eval = self.ddp_model

            # Select DDIM or standard diffusion sampling.
            sample_fn = self.val_diffusion.ddim_sample_loop if self.val_use_ddim else self.val_diffusion.p_sample_loop

            # Prepare the output directory.
            os.makedirs(self.val_save_dir, exist_ok=True)

            # Initialize validation statistics.
            n_sum, n_cnt, saved = 0.0, 0, 0
            seen = 0  # seen samples
            safety_cap = 1000  # Hard limit to prevent unbounded iteration
            max_batches = None

            # Preserve the [-50, 50] display range and multiply by 50 before plotting.
            vis_vmin, vis_vmax = -50.0, 50.0
            denorm = {"cond": 50.0, "pred": 50.0, "gt": 50.0}

            with th.no_grad():
                for bi, batch in enumerate(self.val_loader):
                    # Determine the batch limit from the first validation batch.
                    if max_batches is None:
                        bs = int(batch["image"].size(0))
                        if self.val_max_batches > 0:
                            max_batches = self.val_max_batches
                        elif self.val_len:
                            max_batches = max(1, self.val_len // bs)  # drop_last=True
                        else:
                            max_batches = safety_cap

                    gt = batch["image"].to(dist_util.dev(), non_blocking=True)
                    cond = batch["cond"].to(dist_util.dev(), non_blocking=True)
                    B, _, H, W = gt.shape

                    # Sample predictions.
                    pred = sample_fn(
                        model_for_eval,
                        (B, 1, H, W),
                        clip_denoised=True,
                        model_kwargs={"cond": cond},
                        device=dist_util.dev(),
                    )

                    # Compute batch NRMSE.
                    nrmse_b = compute_nrmse_batch(gt, pred)  # [B]
                    n_sum += float(nrmse_b.sum().item())
                    n_cnt += int(B)
                    seen += int(B)

                    # Rank 0 saves one three-panel figure per sample (cond|pred|gt).
                    is_rank0 = (dist_util.get_world_size() == 1) or (dist_util.get_rank() == 0)
                    if is_rank0 and saved < self.val_num_vis:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        import matplotlib as mpl

                        n_to_save = min(B, self.val_num_vis - saved)

                        # Preserve the original colormap; display NaNs in black.
                        try:
                            cmap = mpl.cm.get_cmap("jet").copy()
                            cmap.set_bad('k')
                        except Exception:
                            cmap = mpl.cm.get_cmap("jet")

                        for k in range(n_to_save):
                            name = batch["name"][k]
                            save_path = os.path.join(self.val_save_dir, f"step_{self.step}_{name}.png")

                            cond_np = (cond[k].detach().cpu().numpy() * denorm["cond"]).squeeze()
                            pred_np = (pred[k].detach().cpu().numpy() * denorm["pred"]).squeeze()
                            gt_np = (gt[k].detach().cpu().numpy() * denorm["gt"]).squeeze()

                            # ======================================
                            # Compute a shared colorbar range dynamically.
                            # ======================================

                            all_min = min(
                                np.nanmin(cond_np),
                                np.nanmin(pred_np),
                                np.nanmin(gt_np),
                            )

                            all_max = max(
                                np.nanmax(cond_np),
                                np.nanmax(pred_np),
                                np.nanmax(gt_np),
                            )

                            if abs(all_max - all_min) < 1e-6:
                                all_max += 1e-6
                                all_min -= 1e-6

                            vis_vmin = float(all_min)
                            vis_vmax = float(all_max)

                            # Draw the three-panel figure.
                            fig, axes = plt.subplots(
                                1, 3, figsize=(12, 4), dpi=200, constrained_layout=True
                            )
                            ims = []
                            ims.append(axes[0].imshow(cond_np, cmap=cmap, vmin=vis_vmin, vmax=vis_vmax,
                                                      origin="upper", aspect="equal"))
                            axes[0].set_title("cond");
                            axes[0].set_xlabel("X");
                            axes[0].set_ylabel("Y")

                            ims.append(axes[1].imshow(pred_np, cmap=cmap, vmin=vis_vmin, vmax=vis_vmax,
                                                      origin="upper", aspect="equal"))
                            axes[1].set_title("pred");
                            axes[1].set_xlabel("X");
                            axes[1].set_ylabel("")

                            ims.append(axes[2].imshow(gt_np, cmap=cmap, vmin=vis_vmin, vmax=vis_vmax,
                                                      origin="upper", aspect="equal"))
                            axes[2].set_title("gt");
                            axes[2].set_xlabel("X");
                            axes[2].set_ylabel("")

                            # Add a shared colorbar on the right.
                            cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
                            cbar.set_label("phase (rad)")

                            fig.suptitle(f"step {self.step} — {name}")
                            fig.savefig(save_path, dpi=200)
                            plt.close(fig)

                        saved += n_to_save

                    # Check stopping conditions.
                    if self.val_max_samples > 0 and seen >= self.val_max_samples:
                        break
                    if (bi + 1) >= max_batches:
                        break

            # Optionally reduce statistics across GPUs.
            if os.getenv("VAL_SKIP_REDUCE", "0") != "1":
                world = dist_util.get_world_size()
                if world > 1 and dist.is_available() and dist.is_initialized():
                    device = dist_util.dev() if th.cuda.is_available() else th.device("cpu")
                    tensor = th.tensor([n_sum, n_cnt], dtype=th.float64, device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    n_sum, n_cnt = tensor.tolist()

            mean_nrmse = (n_sum / max(1.0, n_cnt)) if n_cnt > 0 else float("nan")
            logger.log(f"[VAL] step={self.step}  mean_NRMSE={mean_nrmse:.6f}  images={int(n_cnt)}  (saved={saved})")
            logger.logkv("val/mean_nrmse", mean_nrmse)
            logger.dumpkvs()
            return mean_nrmse

        finally:
            # Restore training weights and return to training mode.
            self.model.load_state_dict(self._master_params_to_state_dict(orig_master))
            self.model.train()

    # ---------- Optimizer / misc ----------

    def optimize_fp16(self):
        # If non-finite grads exist, shrink the loss scale and skip this step
        if any((p.grad is not None) and (not th.isfinite(p.grad).all()) for p in self.model_params):
            self.lg_loss_scale -= 1
            if self.lg_loss_scale <= 0:
                raise FloatingPointError(
                    "FP16 gradients remain non-finite at loss scale <= 1. "
                    "Restart with --use_fp16 False using a finite checkpoint or a fresh run."
                )
            logger.log(f"Non-finite gradients; loss_scale={self.lg_loss_scale:.2f}, update skipped")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        # FP16 master gradients have already been unscaled at this point.
        norm = th.nn.utils.clip_grad_norm_(
            self.master_params,
            self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        logger.logkv_mean("grad_norm", float(norm))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save non-EMA + all EMA checkpoints and optimizer state.
        Only rank 0 writes; barrier at the end if DDP is active.
        """
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist_util.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist_util.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(self.model.parameters(), master_params)
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


# ---------- Misc helpers shared with OpenAI baseline ----------

def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Write scalar/vector losses to logger; skip large tensors to avoid overhead.
    """
    for key, values in losses.items():
        if isinstance(values, th.Tensor) and values.ndim >= 1:
            logger.logkv_mean(key, values.mean().item())
            for sub_t, sub_loss in zip(ts.detach().cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", float(sub_loss))
        else:
            # Non-tensor or scalar tensors:
            try:
                logger.logkv_mean(key, float(values))
            except Exception:
                pass
