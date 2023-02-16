import argparse
import datetime
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BinaryF1Score, Mean
from tqdm import trange

from sat_pref_plan.data.modified import ModifiedPyramidPatchPatchDataset
from sat_pref_plan.models.convnet import ConvClassifier
from sat_pref_plan.run import utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--patchsize", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--numlevels", type=int, default=3)
    parser.add_argument("--nocuda", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nocuda else "cpu"
    )
    logger.info(f"Using {device} device")

    datadir = Path(args.datadir)
    if not datadir.is_dir():
        logger.error(f"Invalid data directory: {datadir}")
        exit(1)

    patch_size: int = args.patchsize
    num_levels: int = args.numlevels
    stride: int = args.stride
    dataset = ModifiedPyramidPatchPatchDataset(
        datadir, patch_size, num_levels=num_levels, stride=stride
    )
    tsize = int(len(dataset) * 0.8)
    vsize = len(dataset) - tsize
    train, val = random_split(
        dataset, [tsize, vsize], generator=torch.Generator().manual_seed(42)
    )
    batch_size: int = args.batchsize
    tloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    vloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Training size: {len(train)}, Validation size: {len(val)}")

    learning_rate = 0.01
    epochs: int = args.epochs
    model = ConvClassifier(1, in_channels=3 * num_levels).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, verbose=True
    )

    stamp = str(datetime.datetime.now()).replace(" ", "_")
    savedir = Path(args.savedir + "/" + stamp)
    savedir.mkdir(parents=True, exist_ok=True)
    ckptdir = savedir.joinpath("ckpts")
    ckptdir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(savedir) + "/tb_logs")

    for t in trange(epochs, desc="Epochs"):
        model.train()
        tl = Mean(device=device)
        tf1 = BinaryF1Score(device=device)
        for X, y in tloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y[:, None].float())
            tf1.update(pred[:, 0], y)
            tl.update(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = tl.compute().item()
        train_f1 = tf1.compute().item()

        torch.save(model.state_dict(), ckptdir.joinpath(f"model_{t}.pt"))

        model.eval()
        with torch.no_grad():
            vf1 = BinaryF1Score(device=device)
            vl = Mean(device=device)
            for X, y in vloader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y[:, None].float())
                vf1.update(pred[:, 0], y)
                vl.update(loss)
            val_loss = vl.compute().item()
            val_f1 = vf1.compute().item()

        tb_writer.add_scalar("Loss/train", train_loss, t)
        tb_writer.add_scalar("F1/train", train_f1, t)
        tb_writer.add_scalar("Loss/val", val_loss, t)
        tb_writer.add_scalar("F1/val", val_f1, t)

        scheduler.step(val_loss)

    utils.saveconfig(savedir, model, args)


if __name__ == "__main__":
    main()
