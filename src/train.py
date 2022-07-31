import numpy as np
import argparse
from models.pl.ageguesser import AgeNetworkPL
from models.torch.ageguesser import AgeNetwork
from dataset.dataloader import AgeDataset

from pathlib import Path
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    
    # fix the seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    torch_model = AgeNetwork(backbone_name="tf_efficientnetv2_b0")

    model = AgeNetworkPL(torch_model, lr = args.lr, ro = args.ro, phi = args.phi)

    dataset_path = Path(args.dataset_path)
    image_dataset_train = AgeDataset(base_path=dataset_path, files_txt=dataset_path / "train.txt", img_size=args.input_size )
    
    dataloader = torch.utils.data.DataLoader(
        image_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        prefetch_factor = 4
    )

    gpus = 1 if torch.cuda.is_available() else 0

    callbacks = [

        ModelCheckpoint(
            monitor="train/w_mae", 
            dirpath=Path(args.log_dir) / "ckpts",
            verbose=True, 
            save_top_k=3, 
            save_last=True,
            mode="min"
        ),

    ]
    
    trainer = pl.Trainer(
        resume_from_checkpoint=Path(args.resume) if args.resume is not None else None,
        max_epochs=args.max_epochs, 
        gpus=gpus,
        callbacks=callbacks,
        default_root_dir=args.log_dir,
        profiler=None,
        num_sanity_val_steps=0,
        precision=16 if gpus == 1 else 32,  # default
        )
    
    trainer.fit(model=model, train_dataloaders=dataloader)

    trainer.logger.finalize()

def get_args_parser():
    parser = argparse.ArgumentParser("AgeGuesser training", add_help=False)
    
    parser.add_argument("--lr", default=0.001, type=float, help="lr")
    parser.add_argument("--ro", default=1.0, type=float, help="ro")
    parser.add_argument("--phi", default=1.25, type=float, help="phi")

    parser.add_argument("--max_epochs", default=100, type=int, help="number of epochs")
    
    parser.add_argument(
        "--log_dir", default="./logs", help="path to save ckpt"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="data path"
    )
    parser.add_argument("--input_size", default=224, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=6, type=int)

    parser.add_argument("--resume", default=None, help="resume from checkpoint")
    
    return parser

if __name__ == "__main__":
    
    args = get_args_parser()
    args = args.parse_args()
    main(args)