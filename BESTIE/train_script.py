import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
jax.config.update("jax_enable_x64", True)
from BESTIE.training.train import Train
import BESTIE

from argparse import ArgumentParser
from tqdm import tqdm

def parser():
    parser = ArgumentParser(description="Train BESTIE model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="unnamed",
        help="Name of the saved model",
    )

    parser.add_argument(
        "--overrides",
        nargs="+",
        type=str,
        default=[],
        help="Overrides for the training configuration",
    )

    parser.add_argument(
        "--pbar",
        action="store_true",
        help="Enables progress bar",
    )

    return parser

def main(config,name,pbar):
    trainer = Train(config,name=name)

    for epoch in tqdm(range(config["training"]["epochs"]),disable=not pbar):
        trainer.train_step(validate=epoch%10==0)

        #checkpoint
        if (epoch+1) % 20 == 0:
            print("Checkpointing...")
            trainer.save_results()
    
    trainer.save_results()

if __name__ == "__main__":
    args = parser().parse_args()
    config = BESTIE.utilities.configs.parse_yaml(args.config)


    for override in args.overrides:
        override_dict = BESTIE.utilities.configs.parse_yaml(override)
        config = BESTIE.utilities.configs.override(config,override_dict)
    main(config,args.name,args.pbar) 
