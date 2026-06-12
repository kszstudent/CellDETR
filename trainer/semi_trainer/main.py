import os
import sys 
from util.config import ConfigDict
from util.distributed import init_distributed_mode
import argparse
from .engine import trainer
import wandb
import copy

def main(cfg:ConfigDict):
    if cfg.experiment.wandb:
        wandb_train = wandb.init(project=cfg.experiment.project,name=cfg.experiment.name)
    else:
        wandb_train = None
    model_trainer = trainer(cfg,wandb_train)
    model_trainer.train()
    print(cfg.model.backbone.name)

if __name__ == "__main__":
    cfg = ConfigDict.from_file('config/experiments/semi_train.yaml')
    init_distributed_mode(cfg)
    main(cfg)