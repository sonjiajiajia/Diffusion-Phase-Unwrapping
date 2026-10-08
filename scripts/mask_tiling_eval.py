#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-guided tiling inference + stitching + saving.

Behavior:
1) Share one mask across all input images: load Mask_Error from results.mat
   once and resize with nearest-neighbor interpolation if needed.
2) Do not filter small connected components.
3) Save full-size and masked prediction TIFFs, corresponding PNGs, and a mask
   PNG. NaNs are transparent in PNGs.

Output variants:
- full: predictions cover all selected tiles, including pixels outside the
  mask within those tiles; pixels outside selected tiles are NaN.
- masked: retain only valid mask pixels and set all other pixels to NaN.

Example (adjust the paths):
python scripts/mask_tiling_eval.py \
  --model_path /path/to/model.pt \
  --cond_dir /path/to/cond_tif_dir \
  --gt_dir /path/to/gt_tif_dir \
  --mask_mat /path/to/results.mat \
  --mask_key Mask_Error \
  --exp_dir /path/to/out_dir \
  --denorm 50 \
  --use_ddim True --timestep_respacing 100 \
  --tile_h 256 --tile_w 256 --overlap 64 \
  --tile_bias_align mean \
  --png_cmap roma --png_dpi 200
"""

import os
import sys
import glob
import csv
import argparse
from typing import Optional, List, Tuple

import numpy as np
import torch as th
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.io import loadmat

# cmcrameri optional
try:
    import cmcrameri.cm as cmc
except Exception:
    cmc = None

# ---- bring in model/diffusion helpers ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# ---------------------------
# IO
# ---------------------------

def list_tif(dirpath: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dirpath, "*.tif"))) + \
           sorted(glob.glob(os.path.join(dirpath, "*.tiff")))

def load_single_channel_tif(path: str) -> np.ndarray:
    arr = tiff.imread(path).astype(np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Common layouts: [1,H,W] or [H,W,1].
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr[0]
    raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

# ---------------------------
# Mask loading & resize
# ---------------------------

def resize_like_nn(src: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    thh, tww = target_hw
    if src.shape == (thh, tww):
        return src
    try:
        import cv2
        return cv2.resize(src.astype(np.float32), (tww, thh), interpolation=cv2.INTER_NEAREST)
    except Exception:
        from skimage.transform import resize as sk_resize
        return sk_resize(
            src.astype(np.float32), (thh, tww),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)

def load_common_mask_from_mat(mat_path: str, key: str) -> np.ndarray:
    d = loadmat(mat_path)
    if key not in d:
        keys = [k for k in d.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {keys}")
    m = np.asarray(d[key])
    # Squeeze singleton dimensions in (H,W,1) or (1,H,W) arrays.
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask '{key}' after squeeze is not 2D, got shape={m.shape}")
    return m.astype(np.float32)

def mask_valid_from_mask_value(mask_val: np.ndarray) -> np.ndarray:
    """
    Mask_Error uses 1 for valid pixels and NaN (or 0) for invalid pixels.
    Treat finite, nonzero values as valid.
    """
    return np.isfinite(mask_val) & (mask_val != 0)

def apply_mask_nan(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[~valid] = np.nan
    return out

# ---------------------------
# Plot (NaN transparent)
# ---------------------------

def _get_cmap(name: str):
    lname = str(name).lower()
    if lname in ("roma", "cmcrameri.roma") and cmc is not None:
        return cmc.roma
    try:
        return mpl.colormaps.get_cmap(name)
    except Exception:
        return mpl.cm.get_cmap(name)

def save_png_nan_transparent(
    img: np.ndarray,
    vmin: float,
    vmax: float,
    out_path: str,
    cmap_name: str = "roma",
    dpi: int = 200,
    title: str = ""
):
    cmap = _get_cmap(cmap_name).copy()
    # NaN -> transparent
    cmap.set_bad((0, 0, 0, 0))

    mimg = np.ma.masked_invalid(img)
    h, w = img.shape
    fig_w = max(4.0, 4.0 * w / 256.0)
    fig_h = max(3.0, 3.0 * h / 256.0)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(mimg, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("value", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)

# ---------------------------
# Fill NaN for model input
# ---------------------------

def fill_nan(arr: np.ndarray, method: str = "nearest") -> np.ndarray:
    if not np.isnan(arr).any():
        return arr
    m = method.lower()
    if m == "zero":
        return np.nan_to_num(arr, nan=0.0)
    if m in ("mean", "median"):
        val = np.nanmean(arr) if m == "mean" else np.nanmedian(arr)
        return np.nan_to_num(arr, nan=float(val))
    if m == "nearest":
        try:
            from scipy import ndimage
            bad = ~np.isfinite(arr)
            if not bad.any():
                return arr
            idx = ndimage.distance_transform_edt(
                bad, return_distances=False, return_indices=True
            )
            return arr[tuple(idx)]
        except Exception:
            return np.nan_to_num(arr, nan=0.0)
    return np.nan_to_num(arr, nan=0.0)

# ---------------------------
# Align (offset)
# ---------------------------

def best_offset(a: np.ndarray, b: np.ndarray, mask: np.ndarray, method: str) -> float:
    """
    Estimate a constant offset so a + offset approximates b within the mask.
    method: none | mean | median | lsq(=mean)
    """
    m = method.lower().strip()
    if m in ("none", "", "off"):
        return 0.0
    mm = mask & np.isfinite(a) & np.isfinite(b)
    diff = (b - a)[mm]
    if diff.size == 0:
        return 0.0
    if m in ("mean", "lsq", "least_squares"):
        return float(np.mean(diff))
    if m == "median":
        return float(np.median(diff))
    raise ValueError(f"Unknown align_method: {method}")

# ---------------------------
# Tiling helpers
# ---------------------------

def _hann2d(h: int, w: int) -> np.ndarray:
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(h) / (h - 1))
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / (w - 1))
    w2d = np.outer(wy, wx).astype(np.float32)
    return np.maximum(w2d, 1e-4)

def _tile_coords(full_h: int, full_w: int, tile_h: int, tile_w: int, overlap: int) -> List[Tuple[int, int]]:
    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    ys = list(range(0, max(1, full_h - tile_h + 1), stride_h))
    xs = list(range(0, max(1, full_w - tile_w + 1), stride_w))
    if ys[-1] != full_h - tile_h:
        ys.append(full_h - tile_h)
    if xs[-1] != full_w - tile_w:
        xs.append(full_w - tile_w)
    return [(y, x) for y in ys for x in xs]

def _select_tiles_by_mask(mask_valid: np.ndarray, tile_h: int, tile_w: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Select tiles containing at least one pixel where mask_valid is True.
    """
    H, W = mask_valid.shape
    coords = _tile_coords(H, W, tile_h, tile_w, overlap)
    keep = []
    for (y, x) in coords:
        if np.any(mask_valid[y:y+tile_h, x:x+tile_w]):
            keep.append((y, x))
    return keep

@th.no_grad()
def _sample_one_patch(sample_fn, model, cond_patch_t: th.Tensor, out_channels: int, device) -> th.Tensor:
    out = sample_fn(
        model,
        (1, out_channels, cond_patch_t.shape[-2], cond_patch_t.shape[-1]),
        clip_denoised=True,
        model_kwargs={"cond": cond_patch_t},
        device=device,
    )
    return out

@th.no_grad()
def tile_infer_mask_guided(
    sample_fn,
    model,
    cond_full_norm: np.ndarray,   # [H,W] model input divided by denorm
    mask_valid: np.ndarray,       # [H,W] boolean validity mask
    out_channels: int,
    tile_hw: Tuple[int, int],
    overlap: int,
    device: th.device,
    tile_bias_align: str = "mean",
    img_name: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - pred_norm_full: [H,W] predictions in selected tiles; NaN elsewhere.
    - tile_covered: [H,W] boolean mask of pixels covered by inferred tiles.
    """
    H, W = cond_full_norm.shape
    thh, tww = tile_hw
    if thh > H or tww > W:
        raise ValueError(f"Tile {thh}x{tww} exceeds image size {H}x{W} for {img_name}")

    # Select tiles.
    coords = _select_tiles_by_mask(mask_valid, thh, tww, overlap)
    if len(coords) == 0:
        pred = np.full((H, W), np.nan, dtype=np.float32)
        covered = np.zeros((H, W), dtype=bool)
        return pred, covered

    acc  = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    win2d = _hann2d(thh, tww)

    prefix = f"[MASK-TILE][{img_name}]" if img_name else "[MASK-TILE]"
    print(f"{prefix} image {H}x{W}, tile {thh}x{tww}, overlap={overlap}, selected_tiles={len(coords)}")

    for i, (y, x) in enumerate(coords):
        cond_patch = cond_full_norm[y:y+thh, x:x+tww]  # [thh,tww]
        cond_t = th.from_numpy(cond_patch[None, None, ...]).to(device)

        pred_patch = _sample_one_patch(sample_fn, model, cond_t, out_channels, device)
        pred_np = pred_patch.detach().cpu().numpy()[0, 0].astype(np.float32)

        # Align overlapping regions using a constant offset.
        mode = tile_bias_align.lower().strip()
        if mode in ("mean", "median"):
            w_exist = wsum[y:y+thh, x:x+tww]
            overlap_m = w_exist > 1e-6
            if np.any(overlap_m):
                existing = acc[y:y+thh, x:x+tww] / np.maximum(w_exist, 1e-6)
                diff = existing[overlap_m] - pred_np[overlap_m]
                if diff.size > 0:
                    delta = float(np.mean(diff)) if mode == "mean" else float(np.median(diff))
                    pred_np += delta

        # Stitch entire tiles; apply the strict mask to the masked output later.
        acc[y:y+thh, x:x+tww]  += pred_np * win2d
        wsum[y:y+thh, x:x+tww] += win2d
        covered[y:y+thh, x:x+tww] = True

        if (i + 1) % 10 == 0 or (i + 1) == len(coords):
            print(f"{prefix} tile {i+1}/{len(coords)} done")

    pred_norm = np.full((H, W), np.nan, dtype=np.float32)
    m = wsum > 1e-6
    pred_norm[m] = acc[m] / wsum[m]
    return pred_norm, covered

# ---------------------------
# Main
# ---------------------------

def find_gt_path(gt_dir: str, base: str) -> Optional[str]:
    cand1 = os.path.join(gt_dir, f"{base}.tif")
    cand2 = os.path.join(gt_dir, f"{base}_unw.tif")
    if os.path.exists(cand1): return cand1
    if os.path.exists(cand2): return cand2
    return None

def create_argparser():
    eval_defaults = dict(
        model_path="",
        use_ddim=True,
        clip_denoised=True,
        timestep_respacing="50",

        cond_dir="",
        gt_dir="",
        exp_dir="./experiment_mask_tiling",

        denorm=50.0,          # Divide inputs by 50 if training used that scale.
        fill_method="nearest",

        align_method="mean",  # none|mean|median

        # mask
        mask_mat="",
        mask_key="Mask_Error",

        # tiling
        tile_h=256,
        tile_w=256,
        overlap=64,
        tile_bias_align="mean",   # none|mean|median

        # saving
        save_tif=True,
        save_png=True,

        # png look
        png_cmap="roma",
        png_dpi=200,

        out_channels=1,
        num_samples=0,  # 0=all
    )

    md_defaults = model_and_diffusion_defaults()
    defaults = {**md_defaults, **eval_defaults}

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()

    # ---- device ----
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, cuda_available={th.cuda.is_available()}")

    # ---- output dirs ----
    exp_dir = args.exp_dir
    tif_dir = os.path.join(exp_dir, "preds_tif")
    png_dir = os.path.join(exp_dir, "preds_png")
    os.makedirs(exp_dir, exist_ok=True)
    if args.save_tif:
        os.makedirs(tif_dir, exist_ok=True)
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)

    # ---- load common mask once ----
    if not args.mask_mat:
        raise ValueError("--mask_mat is required (results.mat path).")
    common_mask_val = load_common_mask_from_mat(args.mask_mat, args.mask_key)
    print(f"[INFO] loaded common mask from {args.mask_mat}, key={args.mask_key}, shape={common_mask_val.shape}")

    # ---- model ----
    print("[INFO] creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    denorm = float(args.denorm) if args.denorm is not None else 0.0
    out_ch = int(args.out_channels)

    # ---- list files ----
    cond_files = list_tif(args.cond_dir)
    if len(cond_files) == 0:
        raise FileNotFoundError(f"No tif/tiff found in cond_dir={args.cond_dir}")
    if args.num_samples and int(args.num_samples) > 0:
        cond_files = cond_files[:int(args.num_samples)]
    print(f"[INFO] cond files: {len(cond_files)}")

    # ---- csv ----
    csv_path = os.path.join(exp_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "name", "H", "W",
            "n_tiles_selected",
            "offset_pred",
        ])

        for idx, cond_path in enumerate(cond_files, start=1):
            base = stem(cond_path)

            # load cond & gt
            cond_raw = load_single_channel_tif(cond_path)           # may contain NaN
            cond_in  = fill_nan(cond_raw.copy(), args.fill_method)  # for model
            H, W = cond_raw.shape

            gt_path = find_gt_path(args.gt_dir, base)
            if gt_path is None:
                print(f"[WARN] GT not found for {base}, skip (need GT for offset alignment).")
                continue
            gg = load_single_channel_tif(gt_path)
            if gg.shape != (H, W):
                print(f"[WARN] GT size mismatch for {base}: gt{gg.shape} vs cond{cond_raw.shape} — resizing GT (linear).")
                try:
                    import cv2
                    gg = cv2.resize(gg.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    from skimage.transform import resize as sk_resize
                    gg = sk_resize(gg.astype(np.float32), (H, W), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

            # resize common mask to this image if needed
            mask_val = common_mask_val if common_mask_val.shape == (H, W) else resize_like_nn(common_mask_val, (H, W))
            mask_valid = mask_valid_from_mask_value(mask_val)

            # --- morphological hole filling ---
            from scipy import ndimage as ndi
            mask_valid = ndi.binary_fill_holes(mask_valid).astype(bool)
            from scipy import ndimage as ndi
            mask_valid = ndi.binary_fill_holes(mask_valid)
            mask_valid = ndi.binary_closing(mask_valid, structure=np.ones((3, 3), dtype=bool)).astype(bool)
            # Restrict alignment and evaluation to finite GT and condition pixels.
            mask_eval = mask_valid & np.isfinite(gg) & np.isfinite(cond_raw)

            # Save a mask PNG for each image, using its basename.
            if args.save_png:
                mask_vis = np.full((H, W), np.nan, dtype=np.float32)
                mask_vis[mask_valid] = 1.0
                save_png_nan_transparent(
                    mask_vis, 0.0, 1.0,
                    os.path.join(png_dir, f"{base}__COMMON_MASK.png"),
                    cmap_name="gray", dpi=int(args.png_dpi),
                    title="COMMON_MASK (1=valid)"
                )

            # cond -> model scale
            cond_norm = (cond_in / denorm) if denorm > 0 else cond_in

            # mask-guided tiling infer
            thh, tww = int(args.tile_h), int(args.tile_w)
            ov = int(args.overlap)

            pred_norm_full, covered = tile_infer_mask_guided(
                sample_fn=sample_fn,
                model=model,
                cond_full_norm=cond_norm,
                mask_valid=mask_valid,
                out_channels=out_ch,
                tile_hw=(thh, tww),
                overlap=ov,
                device=device,
                tile_bias_align=str(args.tile_bias_align),
                img_name=base,
            )

            # back to physical
            pred_full = (pred_norm_full * denorm) if denorm > 0 else pred_norm_full

            # pred_full contains values within selected tiles and NaN elsewhere.
            # Estimate the offset where mask_eval and finite predictions overlap.
            mask_for_offset = mask_eval & np.isfinite(pred_full)

            off_pred = best_offset(pred_full, gg, mask_for_offset, method=str(args.align_method))
            pred_aln_full = pred_full + off_pred

            # Generate the masked output, retaining only valid mask pixels.
            # Final valid pixels must also have a finite condition value.
            final_valid = mask_valid & np.isfinite(cond_raw)

            pred_aln_masked = apply_mask_nan(pred_aln_full, final_valid)
            pred_aln_full = apply_mask_nan(pred_aln_full, np.isfinite(cond_raw))

            # Save TIFF outputs.
            if args.save_tif:
                tiff.imwrite(os.path.join(tif_dir, f"{base}_pred_aln_full.tif"), pred_aln_full.astype(np.float32))
                tiff.imwrite(os.path.join(tif_dir, f"{base}_pred_aln_masked.tif"), pred_aln_masked.astype(np.float32))

            # Save PNGs with transparent NaNs and the GT range within mask_eval.
            if args.save_png:
                if np.any(mask_eval):
                    vals = gg[mask_eval].astype(np.float64)
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                    if vmin == vmax:
                        vmin -= 1e-6
                        vmax += 1e-6
                else:
                    vmin, vmax = -50.0, 50.0

                save_png_nan_transparent(
                    pred_aln_full, vmin, vmax,
                    os.path.join(png_dir, f"{base}__PRED_ALN_FULL.png"),
                    cmap_name=str(args.png_cmap), dpi=int(args.png_dpi),
                    title=f"PRED_ALN_FULL (tile-covered)"
                )
                save_png_nan_transparent(
                    pred_aln_masked, vmin, vmax,
                    os.path.join(png_dir, f"{base}__PRED_ALN_MASKED.png"),
                    cmap_name=str(args.png_cmap), dpi=int(args.png_dpi),
                    title=f"PRED_ALN_MASKED (mask only)"
                )

                # Optionally save masked GT and condition previews for comparison.
                gt_masked = apply_mask_nan(gg, mask_valid)
                cond_masked = apply_mask_nan(cond_raw, mask_valid)
                save_png_nan_transparent(
                    gt_masked, vmin, vmax,
                    os.path.join(png_dir, f"{base}__GT_MASKED.png"),
                    cmap_name=str(args.png_cmap), dpi=int(args.png_dpi),
                    title="GT (masked)"
                )
                save_png_nan_transparent(
                    cond_masked, vmin, vmax,
                    os.path.join(png_dir, f"{base}__COND_MASKED.png"),
                    cmap_name=str(args.png_cmap), dpi=int(args.png_dpi),
                    title="COND (masked)"
                )

            # Write CSV metrics; recount selected coordinates for an exact tile count.
            n_tiles = len(_select_tiles_by_mask(mask_valid, thh, tww, ov))
            writer.writerow([base, H, W, n_tiles, f"{off_pred:.6f}"])

            print(f"[DONE] {idx}/{len(cond_files)} {base}: tiles={n_tiles}, offset={off_pred:.6f}")

    print(f"[INFO] CSV saved to: {csv_path}")
    print("[INFO] done.")

if __name__ == "__main__":
    main()
