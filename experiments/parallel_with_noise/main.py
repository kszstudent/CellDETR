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
    cfg = ConfigDict.from_file('config/experiments/pannuke_noised_boxes.yaml')
    init_distributed_mode(cfg)
    # tuning_dict = [{'experiment':{'name': '312_th04(bt-2-0.6-lr0.0002-80e)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold1'},'test':{'fold':'fold2'}}},
    # {'experiment':{'name': '321-th04(bt-2-0.6-lr0.0002-80e)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold2'},'test':{'fold':'fold1'}}},
    # {'experiment':{'name': '123-th04(bt-2-0.6-lr0.0002-80e)'}, 'dataset':{'train':{'fold':'fold1'}, 'val':{'fold':'fold2'},'test':{'fold':'fold3'}}},]
    tuning_dict = [{'experiment':{'name': '312_th04(bt-3-5:5)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold1'},'test':{'fold':'fold2'}}},
    {'experiment':{'name': '321-th04(bt-3-5:5)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold2'},'test':{'fold':'fold1'}}},
    {'experiment':{'name': '123-th04(bt-3-5:5)'}, 'dataset':{'train':{'fold':'fold1'}, 'val':{'fold':'fold2'},'test':{'fold':'fold3'}}},]
    # tuning_dict = [{'experiment':{'wandb':False}}]
    for tune in tuning_dict:
        cfg_ = copy.deepcopy(cfg)
        cfg_.update(tune)
        dataset = cfg_.dataset
        main(cfg_)
        wandb.finish()
    # main(cfg)