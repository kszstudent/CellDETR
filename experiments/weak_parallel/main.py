import os
from util.config import ConfigDict
from .engine import weak_trainer
import wandb

cfg = ConfigDict.from_file('configs/experiments/decouple/decouple_weak.yaml')
def main():
    if cfg.experiment.wandb:
        wandb_train = wandb.init(project=cfg.experiment.project,name=cfg.experiment.name)
    else:
        wandb_train = None
    model_trainer = weak_trainer(cfg,wandb_train)
    model_trainer.train()
    print(cfg.model.backbone.name)

if __name__ == "__main__":
    main()