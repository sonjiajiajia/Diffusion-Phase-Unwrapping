"""
Train a diffusion model on images.
Now supports --out_dir so checkpoints/logs/val_vis go to one place.
"""

import argparse
import json
import os

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    if args.resume_checkpoint:
        checkpoint_config = os.path.join(
            os.path.dirname(os.path.abspath(args.resume_checkpoint)), "config.json"
        )
        if os.path.isfile(checkpoint_config):
            with open(checkpoint_config) as stream:
                previous_config = json.load(stream)
            if previous_config.get("predict_xstart") != args.predict_xstart:
                raise ValueError("Resume checkpoint prediction mode does not match predict_xstart.")

    dist_util.setup_dist()

    # ---- set unified output root ----
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as stream:
        json.dump(vars(args), stream, indent=2, sort_keys=True)
    # for train_util.get_blob_logdir()
    os.environ["DIFFUSION_BLOB_LOGDIR"] = out_dir
    # for scalar/event logs
    try:
        os.environ.setdefault("OPENAI_LOG_FORMAT", "compact,log,csv")
        logger.configure(dir=out_dir)
    except Exception:
        try:
            logger.set_dir(out_dir)
        except Exception:
            pass

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Train loader (infinite)
    train_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        infinite=True,
    )

    # Val loader (finite, single pass)
    assert args.val_data_dir, "--val_data_dir is required"
    val_loader, val_len = load_data(
        data_dir=args.val_data_dir,
        batch_size=args.val_batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
        infinite=False,
    )

    val_diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.val_timestep_respacing,
    )

    # default val_vis folder = <out_dir>/val_vis (unless user overrode it)
    val_save_dir = args.val_save_dir
    if not val_save_dir:  # None or empty string
        val_save_dir = os.path.join(out_dir, "val_vis")
    elif os.path.normpath(val_save_dir) == os.path.normpath("./val_vis"):
        # keep old default behavior but relocate under out_dir
        val_save_dir = os.path.join(out_dir, "val_vis")
    os.makedirs(val_save_dir, exist_ok=True)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_grad_norm=args.max_grad_norm,

        # validation
        val_loader=val_loader,
        val_len=val_len,
        val_every=args.val_every,
        val_num_vis=args.val_num_vis,
        val_save_dir=val_save_dir,
        val_use_ddim=args.val_use_ddim,
        val_timestep_respacing=args.val_timestep_respacing,
        val_diffusion=val_diffusion,

        # NEW
        val_max_batches=args.val_max_batches,
        val_max_samples=args.val_max_samples,
    ).run_loop()


def create_argparser():
    defaults = dict(
        # unified output dir (checkpoints/logs/val_vis)
        out_dir="./runs/exp1",

        data_dir="",
        val_data_dir="",
        val_batch_size=8,
        val_every=2000,
        val_num_vis=8,
        # None -> put val images under <out_dir>/val_vis
        val_save_dir=None,
        val_use_ddim=True,
        val_timestep_respacing="50",

        # NEW: limit validation workload
        val_max_batches=0,   # 0 = no cap by batches; >0 = validate at most N batches
        val_max_samples=0,   # 0 = no cap by samples; >0 = validate at most N samples

        # existing
        lr=1e-4, weight_decay=0.0, lr_anneal_steps=0, max_grad_norm=1.0,
        batch_size=1, microbatch=-1, ema_rate="0.9999",
        log_interval=10, save_interval=10000, resume_checkpoint="",
        use_fp16=False, fp16_scale_growth=1e-3, schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
