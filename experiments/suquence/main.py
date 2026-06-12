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
    cfg = ConfigDict.from_file('config/experiments/pannuke_supervised.yaml')
    tuning_dict = [{'experiment':{'name': '312_th04(2)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold1'},'test':{'fold':'fold2'}}},
    {'experiment':{'name': '321-th04(2)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold2'},'test':{'fold':'fold1'}}},
    {'experiment':{'name': '123-th04(2)'}, 'dataset':{'train':{'fold':'fold1'}, 'val':{'fold':'fold2'},'test':{'fold':'fold3'}}},]
    init_distributed_mode(cfg)
    for tune in tuning_dict:
        cfg_ = copy.deepcopy(cfg)
        cfg_.update(tune)
        dataset = cfg_.dataset
        main(cfg_)
    # main()