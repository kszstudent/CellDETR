import torch
import torch.multiprocessing as mp
from dataset import build_semi_weak_dataloader
import os
from tqdm import tqdm
import wandb
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
from tqdm import tqdm
import wandb
from models import build_model
from eval.pannuke_eval import *
class weak_trainer():
    def __init__(self, args:dict,wandb_log:wandb):
        self.args = args
        self.device = args.experiment.device
        self.wandb_log = wandb_log
        self.model, self.criterion, self.postprocessors = build_model(args)
        param_dicts = [
        {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.optimizer.lr_backbone,
        },]
        if self.args.experiment.checkpoint!='None':
            checkpoint = torch.load(self.args.experiment.checkpoint,map_location='cuda',weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(args.experiment.device)
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.optimizer.lr,
                                  weight_decay=args.optimizer.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.optimizer.lr_drop)
    
    def train(self):
        for epoch in range(self.args.optimizer.epochs):
            self.model.train()
            self.train_data,self.eval_data = build_semi_weak_dataloader(self.args)
            self.one_epoch_train(self.train_data,epoch)
            self.one_epoch_eval(self.eval_data,epoch)
            self.lr_scheduler.step()
            if epoch>self.args.optimizer.start_saving_epoch and epoch%self.args.optimizer.save_per_epoch == 0:
                self.saving_model(epoch)
            
    def one_epoch_train(self,train_loader,epoch):
        train_tqdm = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{self.args.optimizer.epochs}')
        for _,(img,targets) in enumerate(train_tqdm):
            img = img.to(self.device)
            targets = [{k: to_device(v, self.device) for k, v in t.items()} for t in targets]
            outputs =self.model(img)
            loss_dict = self.criterion(outputs,targets)
            if self.wandb_log is not None:
                self.wandb_log.log({f'train/{k}':v.item() for k,v in loss_dict.items()}) 
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.05)
            self.optimizer.step()
    @torch.no_grad()
    def one_epoch_eval(self,eval_loader,epoch):
        self.model.eval()
        eval_tqdm = tqdm(eval_loader, desc=f'eval Epoch {epoch+1}/{self.args.optimizer.epochs}')
        for _,(img,targets) in enumerate(eval_tqdm):
            img = img.to(self.device)
            targets = [{k: to_device(v, self.device) for k, v in t.items()} for t in targets]
            outputs =self.model(img)
            loss_dict = self.criterion(outputs, targets)
            if self.wandb_log is not None:
                self.wandb_log.log({f'eval/{k}':v.item() for k,v in loss_dict.items()})
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