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
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the dataset configuration file",
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
        if (epoch+1) % 50 == 0:
            print("Checkpointing...")
            trainer.save_results()
    
    trainer.save_results()

if __name__ == "__main__":
    args = parser().parse_args()
    config = BESTIE.utilities.configs.parse_yaml(args.config)

    config["dataset"] = BESTIE.utilities.configs.parse_yaml(args.dataset_config)

    for override in args.overrides:
        override_dict = BESTIE.utilities.configs.parse_yaml(override)
        config = BESTIE.utilities.configs.override(config,override_dict)
    main(config,args.name,args.pbar) 
