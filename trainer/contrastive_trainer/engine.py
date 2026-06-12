import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import (CosineAnnealingLR,SequentialLR,LinearLR)
from dataset import build_semi_weak_dataloader
from models import build_model
from eval.pannuke_eval import *
class trainer():
    def __init__(self, args:dict,wandb_log:wandb):
        self.args = args
        self.device = args.experiment.device
        self.wandb_log = wandb_log
        self.model, self.criterion, self.postprocessor = build_model(args)
        param_dicts = [
        {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.optimizer.lr_backbone,
        },]
        if self.args.experiment.checkpoint!='None':
            checkpoint = torch.load(self.args.experiment.checkpoint,map_location='cuda',weights_only=False)
            if self.args.experiment.fintune:
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict,strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(args.experiment.device)
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.optimizer.lr,
                                  weight_decay=args.optimizer.weight_decay)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.optimizer.lr_drop)
        # self.test_dataset = build_dataset(self.args, split='test')
        # self.test_loader = build_loader(self.args, self.test_dataset,split='test')
    
    def train(self):
        train_loader,val_loader = build_semi_weak_dataloader(self.args)
        num_warmup_steps = self.args.optimizer.num_warmup_steps
        num_training_steps = len(train_loader) * self.args.optimizer.epochs
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps - num_warmup_steps)
        self.lr_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        for epoch in range(self.args.optimizer.epochs):
            self.model.train()
            self.one_epoch_train(train_loader,epoch)
            self.one_epoch_eval(val_loader,epoch)
            self.lr_scheduler.step()
            if epoch>self.args.optimizer.start_saving_epoch and (epoch+1)%self.args.optimizer.save_per_epoch == 0:
                self.saving_model(epoch)
            
    def one_epoch_train(self,train_loader,epoch):
        train_tqdm = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{self.args.optimizer.epochs}')
        for img,targets in train_tqdm:
            img = img.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs =self.model(img,targets,stage='train')
            loss_dict = self.criterion(outputs,targets)
            if self.wandb_log is not None:
                log_dict = {f'train/{k}':v.item() for k,v in loss_dict.items()}
                log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                self.wandb_log.log(log_dict) 
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()
    @torch.no_grad()
    def one_epoch_eval(self,eval_loader,epoch):
        self.model.eval()
        eval_tqdm = tqdm(eval_loader, desc=f'eval Epoch {epoch+1}/{self.args.optimizer.epochs}')
        epoch_loss_list = []
        for img,targets in eval_tqdm:
            img = img.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs =self.model(img,targets,stage='eval')
            loss_dict = self.criterion(outputs, targets)
            epoch_loss_list.append(loss_dict)
        epoch_loss_dict = {}
        for d in epoch_loss_list:
            for k,v in d.items():
                epoch_loss_dict[k] = epoch_loss_dict.get(k,0.0) + v
        epoch_loss_dict = {k: v / len(epoch_loss_list) for k,v in epoch_loss_dict.items()}
        if self.wandb_log is not None:
            self.wandb_log.log({f'eval/{k}':v.item() for k,v in epoch_loss_dict.items()})
    def saving_model(self,epoch):
        saving_path  = os.path.join(self.args.experiment.output_dir,self.args.experiment.name)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            print(f"model path:{saving_path} created")
        model_name = f'epoch_{epoch}.pth'
        model_path = os.path.join(saving_path,model_name)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss':{'train_loss':self.wandb_log.summary.get('train/total_loss', 0.0),
                    'eval_loss':self.wandb_log.summary.get('eval/total_loss', 0.0)},
            'config': self.args,
            }
        torch.save(checkpoint, model_path)
        print(f'{model_name} saved at {model_path}')
def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    elif isinstance(item, str):
        return item
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))
def map_tensor_values(tensor, mapping_dict):
    """
    according to the mapping_dict, map the values in tensor to new values.
    """
    result = tensor.clone()  # 创建tensor的副本以避免修改原始tensor
    for old_val, new_val in mapping_dict.items():
        result[torch.where(result == old_val)] = new_val
    return result