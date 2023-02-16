import argparse
import json
from pathlib import Path

import torch


def saveconfig(savedir: Path, model: torch.nn.Module, args: argparse.Namespace) -> None:
    config = {
        "model": model.__class__.__name__,
        "datadir": args.datadir,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "patchsize": args.patchsize,
        "numlevels": args.numlevels,
        "stride": args.stride,
    }
    with open(savedir.joinpath("config.txt"), "w") as f:
        json.dump(config, f, indent=4)
