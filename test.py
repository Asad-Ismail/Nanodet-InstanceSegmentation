
import argparse
import os
import warnings

import pytorch_lightning as pl
import torch

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset


from nanodet.util import (
    cfg,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

args = parse_args()
load_config(cfg, args.config)

if cfg.model.arch.head.num_classes != len(cfg.class_names):
    raise ValueError(
        "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
        "but got {} and {}".format(cfg.model.arch.head.num_classes, len(cfg.class_names)))

print("Setting up data...")
train_dataset = build_dataset(cfg.data.train, "train",class_names=cfg.class_names)
#val_dataset = build_dataset(cfg.data.val, "test")

