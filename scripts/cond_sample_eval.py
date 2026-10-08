#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import csv
from typing import Optional, List, Tuple
import sys

import numpy as np
import torch as th
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# =========================
# IO
# =========================

def list_tif(dirpath: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dirpath, "*.tif"))) + \
           sorted(glob.glob(os.path.join(dirpath, "*.tiff")))


def list_cond_files(cond_dir: str) -> List[str]:
    files = list_tif(cond_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .tif/.tiff files in {cond_dir}")
    return files


def load_single_channel_tif(path: str) -> np.ndarray:
    arr = tiff.imread(path).astype(np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr[0]
    raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")


def find_gt_path(gt_dir: str, basename: str) -> Optional[str]:
    candidates = [
        os.path.join(gt_dir, f"{basename}.tif"),
        os.path.join(gt_dir, f"{basename}.tiff"),
        os.path.join(gt_dir, f"{basename}_unw.tif"),
        os.path.join(gt_dir, f"{basename}_unw.tiff"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_wrap_path(wrap_dir: str, basename: str) -> Optional[str]:
    if not wrap_dir:
        return None

    candidates = [
        os.path.join(wrap_dir, f"{basename}.tif"),
        os.path.join(wrap_dir, f"{basename}.tiff"),
        os.path.join(wrap_dir, f"{basename}.geo.diff_pha.tif"),
        os.path.join(wrap_dir, f"{basename}.geo.diff_pha.tiff"),
    ]

    if basename.endswith(".geo.unw"):
        base = basename.replace(".geo.unw", "")
        candidates += [
            os.path.join(wrap_dir, f"{base}.geo.diff_pha.tif"),
            os.path.join(wrap_dir, f"{base}.geo.diff_pha.tiff"),
        ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def fill_nan(arr: np.ndarray, method: str = "nearest") -> np.ndarray:
    if not np.isnan(arr).any():
        return arr.astype(np.float32)

    method = method.lower()

    if method == "zero":
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)

    if method == "mean":
        return np.nan_to_num(arr, nan=float(np.nanmean(arr))).astype(np.float32)

    if method == "median":
        return np.nan_to_num(arr, nan=float(np.nanmedian(arr))).astype(np.float32)

    if method == "nearest":
        try:
            from scipy import ndimage
            bad = ~np.isfinite(arr)
            idx = ndimage.distance_transform_edt(
                bad,
                return_distances=False,
                return_indices=True,
            )
            return arr[tuple(idx)].astype(np.float32)
        except Exception:
            return np.nan_to_num(arr, nan=0.0).astype(np.float32)

    return np.nan_to_num(arr, nan=0.0).astype(np.float32)


# =========================
# Metrics
# =========================

def best_offset(a: np.ndarray, b: np.ndarray, mask: np.ndarray, method: str) -> float:
    method = method.lower()
    if method in ("none", "off", ""):
        return 0.0

    m = mask & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) == 0:
        return 0.0

    diff = b[m] - a[m]

    if method in ("mean", "lsq", "least_squares"):
        return float(np.mean(diff))
    if method == "median":
        return float(np.median(diff))

    raise ValueError(f"Unknown align_method: {method}")


def nrmse_ratio(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)

    m = mask & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) == 0:
        return float("nan")

    aa = a[m].astype(np.float64)
    bb = b[m].astype(np.float64)

    rng = float(np.max(bb) - np.min(bb))
    if rng < 1e-8:
        return float("nan")

    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    return rmse / rng


def ssim_global(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)

    m = mask & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) < 32:
        return float("nan")

    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)

    L = float(np.max(y) - np.min(y))
    if L < 1e-8:
        return float("nan")

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ux = float(np.mean(x))
    uy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - ux) * (y - uy)))

    return float(((2 * ux * uy + C1) * (2 * cov + C2)) /
                 ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)))


def build_eval_mask(gt: np.ndarray, cond: np.ndarray, pred: Optional[np.ndarray], mode: str) -> np.ndarray:
    g = np.isfinite(gt)
    c = np.isfinite(cond)
    p = np.ones_like(g, dtype=bool) if pred is None else np.isfinite(pred)

    mode = mode.lower()
    if mode == "gt":
        return g
    if mode == "gt_cond":
        return g & c
    if mode == "all":
        return g & c & p
    if mode == "holes":
        return g & (~c)

    raise ValueError(f"Unknown mask_mode={mode}")


# =========================
# Visualization
# =========================

def resize_for_png(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    if max_size is None or max_size <= 0:
        return img

    h, w = img.shape
    scale = min(max_size / h, max_size / w, 1.0)
    if scale >= 1.0:
        return img

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize
        return resize(
            img.astype(np.float32),
            (new_h, new_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)


def get_cmap(name: str):
    try:
        cmap = mpl.colormaps.get_cmap(name).copy()
    except Exception:
        cmap = mpl.cm.get_cmap(name).copy()
    cmap.set_bad("k")
    return cmap


def save_composite(cond, pred, gt, wrap, path, cmap_name="jet", dpi=150, png_max_size=1024):
    cmap = get_cmap(cmap_name)

    cond = resize_for_png(cond, png_max_size)
    pred = resize_for_png(pred, png_max_size)
    gt = resize_for_png(gt, png_max_size)
    wrap = resize_for_png(wrap, png_max_size)

    vals = np.concatenate([
        cond[np.isfinite(cond)].ravel(),
        pred[np.isfinite(pred)].ravel(),
        gt[np.isfinite(gt)].ravel(),
    ])

    if vals.size == 0:
        vmin, vmax = -50.0, 50.0
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if abs(vmax - vmin) < 1e-8:
            vmin -= 1e-6
            vmax += 1e-6

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=dpi)

    im0 = axes[0].imshow(wrap, cmap=cmap, vmin=-np.pi, vmax=np.pi, origin="upper")
    axes[0].set_title("WRAP")

    im1 = axes[1].imshow(cond, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title("COND")

    axes[2].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axes[2].set_title("PRED")

    axes[3].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axes[3].set_title("GT")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(im1, ax=axes[1:].ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("phase (rad)")

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("wrapped phase (rad)")

    fig.subplots_adjust(left=0.04, right=0.95, top=0.88, bottom=0.10, wspace=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# =========================
# Tiling
# =========================

def hann2d(h: int, w: int) -> np.ndarray:
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(h) / (h - 1))
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / (w - 1))
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 1e-4)


def tile_coords(H: int, W: int, tile_h: int, tile_w: int, overlap: int) -> List[Tuple[int, int]]:
    if tile_h > H or tile_w > W:
        raise ValueError(f"Tile {tile_h}x{tile_w} exceeds image size {H}x{W}")

    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    ys = list(range(0, max(1, H - tile_h + 1), stride_h))
    xs = list(range(0, max(1, W - tile_w + 1), stride_w))

    if ys[-1] != H - tile_h:
        ys.append(H - tile_h)
    if xs[-1] != W - tile_w:
        xs.append(W - tile_w)

    return [(y, x) for y in ys for x in xs]


@th.no_grad()
def sample_patch(sample_fn, model, cond_t, out_channels, device, clip_denoised=True):
    return sample_fn(
        model,
        (1, out_channels, cond_t.shape[-2], cond_t.shape[-1]),
        clip_denoised=clip_denoised,
        model_kwargs={"cond": cond_t},
        device=device,
    )


@th.no_grad()
def tile_infer(
    sample_fn,
    model,
    cond_norm: np.ndarray,
    out_channels: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    device,
    tile_bias_align: str = "mean",
    clip_denoised=True,
) -> np.ndarray:

    H, W = cond_norm.shape
    coords = tile_coords(H, W, tile_h, tile_w, overlap)

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    win = hann2d(tile_h, tile_w)

    print(f"[TILE] image={H}x{W}, tile={tile_h}x{tile_w}, overlap={overlap}, n_tiles={len(coords)}")

    for i, (y, x) in enumerate(coords):
        patch = cond_norm[y:y + tile_h, x:x + tile_w]
        cond_t = th.from_numpy(patch[None, None, ...]).to(device)

        pred_t = sample_patch(
            sample_fn,
            model,
            cond_t,
            out_channels,
            device,
            clip_denoised=clip_denoised,
        )

        pred_np = pred_t.detach().cpu().numpy()[0, 0]

        if tile_bias_align.lower() not in ("none", "off", ""):
            old_w = wsum[y:y + tile_h, x:x + tile_w]
            overlap_mask = old_w > 1e-6
            if np.any(overlap_mask):
                old_pred = acc[y:y + tile_h, x:x + tile_w] / np.maximum(old_w, 1e-6)
                diff = old_pred[overlap_mask] - pred_np[overlap_mask]
                if diff.size > 0:
                    if tile_bias_align.lower() == "median":
                        bias = float(np.median(diff))
                    else:
                        bias = float(np.mean(diff))
                    pred_np += bias

        acc[y:y + tile_h, x:x + tile_w] += pred_np * win
        wsum[y:y + tile_h, x:x + tile_w] += win

        print(f"[TILE] {i + 1}/{len(coords)} done")

    return acc / np.maximum(wsum, 1e-6)


# =========================
# Main
# =========================

def main():
    args = create_argparser().parse_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    os.makedirs(args.exp_dir, exist_ok=True)
    comp_dir = os.path.join(args.exp_dir, "composites")
    tif_dir = os.path.join(args.exp_dir, "preds_tif")
    os.makedirs(comp_dir, exist_ok=True)
    if args.save_tif:
        os.makedirs(tif_dir, exist_ok=True)

    print("[INFO] creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    state = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    cond_files = list_cond_files(args.cond_dir)
    if args.num_samples > 0:
        cond_files = cond_files[:args.num_samples]

    print(f"[INFO] found {len(cond_files)} cond files")
    print(f"[INFO] exp_dir={args.exp_dir}")

    csv_path = os.path.join(args.exp_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["name", "H", "W", "offset_pred", "nrmse_pred_pct", "ssim_pred", "mae_pred"])

        for cond_path in cond_files:
            base = os.path.splitext(os.path.basename(cond_path))[0]
            print(f"\n========== Processing {base} ==========")

            cond_raw = load_single_channel_tif(cond_path)
            cond_in = fill_nan(cond_raw.copy(), args.fill_method)

            H, W = cond_raw.shape

            if args.image_size is not None and int(args.image_size) > 0 and not args.use_tiling:
                if (H, W) != (args.image_size, args.image_size):
                    raise ValueError(
                        f"Cond size {H}x{W} != image_size {args.image_size}. "
                        f"Use --image_size 0 or --use_tiling True."
                    )

            denorm = float(args.denorm)
            cond_norm = cond_in / denorm if denorm > 0 else cond_in

            if args.use_tiling:
                pred_norm = tile_infer(
                    sample_fn=sample_fn,
                    model=model,
                    cond_norm=cond_norm,
                    out_channels=int(args.out_channels),
                    tile_h=int(args.tile_h),
                    tile_w=int(args.tile_w),
                    overlap=int(args.overlap),
                    device=device,
                    tile_bias_align=args.tile_bias_align,
                    clip_denoised=args.clip_denoised,
                )
            else:
                cond_t = th.from_numpy(cond_norm[None, None, ...]).to(device)
                with th.no_grad():
                    pred_t = sample_fn(
                        model,
                        (1, int(args.out_channels), H, W),
                        clip_denoised=args.clip_denoised,
                        model_kwargs={"cond": cond_t},
                        device=device,
                    )
                pred_norm = pred_t.detach().cpu().numpy()[0, 0]

            pred = pred_norm * denorm if denorm > 0 else pred_norm

            gt_path = find_gt_path(args.gt_dir, base) if args.gt_dir else None
            if gt_path is not None:
                gt = load_single_channel_tif(gt_path)
                mask = build_eval_mask(gt, cond_raw, pred, args.mask_mode)

                off = best_offset(pred, gt, mask, args.align_method)
                pred_aln = pred + off

                r = nrmse_ratio(pred_aln, gt, mask)
                nrmse_pct = r * 100.0 if np.isfinite(r) else np.nan
                ssim_val = ssim_global(pred_aln, gt, mask)

                err = np.abs(pred_aln - gt)
                valid = mask & np.isfinite(err)
                mae = float(np.mean(err[valid])) if np.any(valid) else float("nan")

                writer.writerow([
                    base, H, W,
                    f"{off:.6f}",
                    f"{nrmse_pct:.3f}" if np.isfinite(nrmse_pct) else "nan",
                    f"{ssim_val:.6f}" if np.isfinite(ssim_val) else "nan",
                    f"{mae:.6f}" if np.isfinite(mae) else "nan",
                ])

                pred_save = pred_aln.astype(np.float32)

                wrap_path = find_wrap_path(args.wrap_dir, base) if args.wrap_dir else None
                if wrap_path is not None:
                    wrap = load_single_channel_tif(wrap_path)
                    if wrap.shape != gt.shape:
                        wrap = np.angle(np.exp(1j * gt)).astype(np.float32)
                else:
                    wrap = np.angle(np.exp(1j * gt)).astype(np.float32)

                comp_path = os.path.join(comp_dir, f"{base}.png")
                save_composite(
                    cond=cond_raw,
                    pred=pred_save,
                    gt=gt,
                    wrap=wrap,
                    path=comp_path,
                    cmap_name=args.png_cmap,
                    dpi=int(args.png_dpi),
                    png_max_size=int(args.png_max_size),
                )

                print(f"[METRIC] NRMSE={nrmse_pct:.3f}%  SSIM={ssim_val:.4f}  MAE={mae:.4f}")

            else:
                print(f"[WARN] no GT found for {base}; save prediction only")
                pred_save = pred.astype(np.float32)

            if args.save_tif:
                out_tif = os.path.join(tif_dir, f"{base}_pred.tif")
                tiff.imwrite(out_tif, pred_save.astype(np.float32))
                print(f"[SAVE] {out_tif}")

    print(f"\n[INFO] CSV saved to: {csv_path}")
    print("[INFO] sampling complete")


def create_argparser():
    eval_defaults = dict(
        model_path="",
        cond_dir="",
        gt_dir="",
        wrap_dir="",
        exp_dir="./experiment_eval",

        use_ddim=True,
        clip_denoised=True,
        timestep_respacing="100",

        image_size=0,
        batch_size=1,
        num_samples=0,

        denorm=50.0,
        align_method="median",

        fill_method="nearest",
        mask_mode="gt_cond",

        save_tif=True,

        use_tiling=True,
        tile_h=512,
        tile_w=512,
        overlap=128,
        tile_bias_align="mean",

        png_cmap="jet",
        png_dpi=120,
        png_max_size=512,

        out_channels=1,
    )

    defaults = {**model_and_diffusion_defaults(), **eval_defaults}

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
