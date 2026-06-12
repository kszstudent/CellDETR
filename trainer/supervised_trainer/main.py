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
    ##basic train
    tuning_dict = [{'experiment':{'name': '312_th04(cls_50)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold1'},'test':{'fold':'fold2'}}},
    {'experiment':{'name': '321-th04(cls_50)'}, 'dataset':{'train':{'fold':'fold3'}, 'val':{'fold':'fold2'},'test':{'fold':'fold1'}}},
    {'experiment':{'name': '123-th04(cls_50)'}, 'dataset':{'train':{'fold':'fold1'}, 'val':{'fold':'fold2'},'test':{'fold':'fold3'}}},]
    ##tuning backbone lr
    # tuning_dict = [{'experiment':{'name': 'hp_backbone_lr'}, 'model':{'backbone':{'lr':1e-4}}},
    # {'experiment':{'name': 'hp_backbone_lr_1'}, 'model':{'backbone':{'lr':1e-5}}},
    # {'experiment':{'name': 'hp_backbone_lr_2'}, 'model':{'backbone':{'lr':1e-6}}},]
    ##tuning warmup steps
    # tuning_dict = [{'experiment':{'name': 'hp_warmup'}, 'optimizer':{'num_warmup_steps':500}},
    # {'experiment':{'name': 'hp_warmup_1'}, 'optimizer':{'num_warmup_steps':1000}},
    # {'experiment':{'name': 'hp_warmup_2'}, 'optimizer':{'num_warmup_steps':2000}},
    # {'experiment':{'name': 'hp_warmup_3'}, 'optimizer':{'num_warmup_steps':3000}},
    # {'experiment':{'name': 'hp_warmup_4'}, 'optimizer':{'num_warmup_steps':4000}},]
    ##tuning det loss coef
    # tuning_dict = [{'experiment':{'name': 'hp_det'}, 'loss':{'class_loss_coef':1.5}},
    # {'experiment':{'name': 'hp_det_1'}, 'loss':{'class_loss_coef':2}},
    # {'experiment':{'name': 'hp_det_2'}, 'loss':{'class_loss_coef':2.5}},]
    ##tuning batch size
    # tuning_dict = [{'experiment':{'name': 'hp_bs'}, 'loader':{'train':{'batch_size':4}}},
    # {'experiment':{'name': 'hp_bs_1'}, 'loader':{'train':{'batch_size':8}}},
    # {'experiment':{'name': 'hp_bs_2'}, 'loader':{'train':{'batch_size':32}}},]
    ##tuning query number
    # tuning_dict = [{'experiment':{'name': 'hp_querynum'}, 'model':{'num_queries':500}},
    # {'experiment':{'name': 'hp_querynum_1'}, 'model':{'num_queries':600}},
    # {'experiment':{'name': 'hp_querynum_2'}, 'model':{'num_queries':700}},]
    init_distributed_mode(cfg)
    for tune in tuning_dict:
        cfg_ = copy.deepcopy(cfg)
        cfg_.update(tune)
        print(cfg_)
        dataset = cfg_.dataset
        main(cfg_)
        wandb.finish()
    # main()