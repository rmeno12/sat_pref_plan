import argparse
import json
from pathlib import Path

import torch

from sat_pref_plan.data.base import PatchDataset


def saveconfig(
    savedir: Path,
    model: torch.nn.Module,
    dataset: PatchDataset,
    args: argparse.Namespace,
) -> None:
    config = {
        "model": model.__class__.__name__,
        "datadir": args.datadir,
        "dataset": dataset.__class__.__name__,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "patchsize": args.patchsize,
        "stride": args.stride,
        "numlevels": args.numlevels if hasattr(args, "numlevels") else None,
    }
    with open(savedir.joinpath("config.txt"), "w") as f:
        json.dump(config, f, indent=4)
