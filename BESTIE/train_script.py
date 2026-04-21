import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import sys
import subprocess
import tempfile
import argparse
import yaml
from tqdm import tqdm


def build_parser():
    p = argparse.ArgumentParser(description="Train BESTIE model")
    p.add_argument("--config", type=str, required=True,
                   help="Path to the training configuration file")
    p.add_argument("--name", type=str, default="unnamed",
                   help="Name of the saved model")
    p.add_argument("--overrides", nargs="+", type=str, default=[],
                   help="Override config files applied in order")
    p.add_argument("--pbar", action="store_true",
                   help="Enable progress bar")
    p.add_argument("--float64", action="store_true",
                   help="Enable float64 precision")
    p.add_argument("--warmup_on_float64", action="store_true",
                   help="Train for --warmup_epochs in float64, save result_warmup.pickle, "
                        "then restart in float32 from those weights")
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="Number of float64 warmup epochs (default: 10)")
    p.add_argument("--warmup_checkpoint", type=str, default=None,
                   help="Path to a result.pickle to use as float32 warm-start "
                        "(skips float64 warmup)")
    return p


def load_config(args):
    import BESTIE
    config = BESTIE.utilities.configs.parse_yaml(args.config)
    for override in args.overrides:
        config = BESTIE.utilities.configs.override(
            config, BESTIE.utilities.configs.parse_yaml(override))
    return config


def run_training(trainer, config, pbar):
    num_epochs = config["training"]["epochs"]
    for epoch in tqdm(range(num_epochs), disable=not pbar):
        print(f"Epoch {epoch}")
        trainer.train_step(validate=epoch % 5 == 0)
        if (epoch + 1) % 5 == 0:
            print("Checkpointing...")
            trainer.save_results()
    trainer.save_results()


def run_float64_warmup(args, warmup_dir, warmup_epochs):
    """Spawn a float64 subprocess that trains for warmup_epochs into warmup_dir."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "training": {"epochs": warmup_epochs},
            "save_dir": warmup_dir,
        }, f)
        epochs_override = f.name

    # Collect all overrides: user-supplied ones first, then the warmup override
    all_overrides = args.overrides + [epochs_override]

    cmd = (
        [sys.executable, __file__,
         "--config", args.config,
         "--name", f"{args.name}_warmup",
         "--float64",
         "--overrides"] + all_overrides
        + (["--pbar"] if args.pbar else [])
    )

    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(epochs_override)


def main():
    args = build_parser().parse_args()

    # Must be set before any JAX computation is triggered
    if args.float64:
        jax.config.update("jax_enable_x64", True)

    from BESTIE.training.train import Train
    config = load_config(args)

    if args.warmup_on_float64:
        # ── Phase 1: float64 warmup ──────────────────────────────────────────
        warmup_dir = os.path.join(config["output_dir"], f"{args.name}_warmup")
        os.makedirs(warmup_dir, exist_ok=True)
        warmup_checkpoint = os.path.join(warmup_dir, "result.pickle")

        print(f"=== Phase 1: float64 warmup for {args.warmup_epochs} epochs ===")
        run_float64_warmup(args, warmup_dir, args.warmup_epochs)

        # ── Phase 2: float32 from warmup weights ─────────────────────────────
        print("=== Phase 2: float32 training from warmup checkpoint ===")
        pretrained_params = Train.load_checkpoint_params(warmup_checkpoint)
        trainer = Train(config, name=args.name, pretrained_params=pretrained_params)
        run_training(trainer, config, args.pbar)

    else:
        # Normal run — optionally warm-started from an existing checkpoint
        pretrained_params = None
        if args.warmup_checkpoint:
            print(f"Loading warmup checkpoint from {args.warmup_checkpoint}")
            pretrained_params = Train.load_checkpoint_params(args.warmup_checkpoint)
        trainer = Train(config, name=args.name, pretrained_params=pretrained_params)
        run_training(trainer, config, args.pbar)


if __name__ == "__main__":
    main()
