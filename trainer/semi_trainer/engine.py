import torch
import torch.multiprocessing as mp
from util.config import ConfigDict
from torch.optim.lr_scheduler import (CosineAnnealingLR,SequentialLR,LinearLR)
mp.set_sharing_strategy('file_system')
import os
from tqdm import tqdm
import wandb
from dataset import (build_dataset, build_loader,build_semi_weak_dataloader)
from models import build_model
from eval.pannuke_eval import *
CELL_LIST = ['T','Myeloid','Malignant','NK','Mast','Fibroblast','Epithelial','Endothelial','B','Plasma']
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
        self.test_dataset = build_dataset(self.args, split='test')
        self.test_loader = build_loader(self.args, self.test_dataset,split='test')
        cfg = ConfigDict.from_file('config/experiments/semi_train.yaml')
        cfg.dataset.root = '/public/home/zhangshikang/project/DATA/semi_data/split/eval'
        cfg.dataset.train_pct = 1.0
        self.test_loader_semi,_ = build_semi_weak_dataloader(cfg)
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
            self.test()
            self.test_on_xenium()
            # self.lr_scheduler.step()
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
        for img,targets in eval_tqdm:
            img = img.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs =self.model(img,targets,stage='eval')
            loss_dict = self.criterion(outputs, targets)
            if self.wandb_log is not None:
                self.wandb_log.log({f'eval/{k}':v.item() for k,v in loss_dict.items()})

    @torch.no_grad()
    def test(self):
        metrics = {
        #'map' : torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh'),
        'f'   : CellDetectionMetric(num_classes=5, 
                                    thresholds=0.4,
                                    max_pair_distance=12,
                                    class_names=self.test_dataset.class_names)
            }
        map_dict = {0: 1,1: 1,2: 0,3: 1,4: 0,5: 2,6: 4,7: 2,8: 1,9: 1,10:2,11:2,12:1
        }
        ### version 2
        model = self.model.to('cuda')
        for imgs,targets in tqdm(self.test_loader):
            imgs = imgs.tensors.to('cuda')
            model.eval()
            outputs_ = {}
            with torch.no_grad():
                outputs = model(imgs,targets,stage='test',threshold=0.4)
                instance_logits = outputs['decouple_class_logits']
                instance_length = outputs['instance_length']
                instance_boxes = outputs['instance_boxes']
                for i, l in enumerate(instance_length):
                    instance_boxes[i,l:,:2] = 2
                    instance_logits[i,l:,:] = -100 
            outputs_['pred_logits'] = instance_logits.clone().cpu()
            outputs_['pred_boxes'] = instance_boxes.clone().cpu()
            orig_target_sizes = torch.stack([torch.tensor(t["boxes"].canvas_size) for t in targets], dim=0)
            predictions = self.postprocessor['bbox'](outputs_, orig_target_sizes)
            for p in predictions:
                # convert boxes
                p['boxes'] = box_ops.box_xyxy_to_cxcywh(p['boxes'])
                
                p['labels'] = map_tensor_values(p['labels'], map_dict)
            # prepare targets
            for t in targets:
                # get image size
                #img_h, img_w = data_loader.dataset.image_size(image_id=t["image_id"])
                img_h, img_w = t['boxes'].canvas_size
                # convert boxes
                t['boxes'] = box_ops.denormalize_box(t['boxes'], (img_h, img_w))
            # update metrics
            for k in metrics:
                metrics[k].update(predictions, targets)
        metrics = {k: metrics[k].compute() for k in metrics}
        flattened_metrics = {}
        for outer_key, outer_value in metrics.items():
            for inner_key, inner_value in outer_value.items():
                for class_name, class_metrics in inner_value.items():
                    for metric_name, metric_value in class_metrics.items():
                        wandb_key = f"{outer_key}/{inner_key}/{class_name}/{metric_name}"
                        flattened_metrics[wandb_key] = metric_value
        if self.wandb_log is not None:
            self.wandb_log.log(flattened_metrics)
    def test_on_xenium(self):
        metrics = {
        #'map' : torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh'),
        'f'   : CellDetectionMetric(num_classes=10, 
                                    thresholds=0.4,
                                    max_pair_distance=12,
                                    class_names=CELL_LIST)
            }
        ### version 2
        model = self.model.to('cuda')
        for imgs,targets in tqdm(self.test_loader_semi):
            imgs = imgs.to('cuda')
            for t in targets:
                labels = t['labels']
                t['boxes'] = t['boxes'][torch.where(labels!=-1)]
                t['labels'] = labels[torch.where(labels!=-1)]
            model.eval()
            outputs_ = {}
            with torch.no_grad():
                outputs = model(imgs,targets,stage='test',threshold=0.4)
                instance_logits = outputs['decouple_class_logits']
                instance_length = outputs['instance_length']
                instance_boxes = outputs['instance_boxes']
                for i, l in enumerate(instance_length):
                    instance_boxes[i,l:,:2] = 2
                    instance_logits[i,l:,:] = -100 
            outputs_['pred_logits'] = instance_logits.clone().cpu()
            outputs_['pred_boxes'] = instance_boxes.clone().cpu()
            orig_target_sizes = torch.stack([torch.tensor([256,256]) for t in targets], dim=0)
            predictions = self.postprocessor['bbox'](outputs_, orig_target_sizes)
            for p in predictions:
                # convert boxes
                p['boxes'] = box_ops.box_xyxy_to_cxcywh(p['boxes'])
            # prepare targets
            for t in targets:
                # get image size
                #img_h, img_w = data_loader.dataset.image_size(image_id=t["image_id"])
                img_h, img_w = 256,256
                # convert boxes
                t['boxes'] = box_ops.denormalize_box(t['boxes'], (img_h, img_w))
            # update metrics
            for k in metrics:
                metrics[k].update(predictions, targets)
        metrics = {k: metrics[k].compute() for k in metrics}
        metrics_ = {}
        metrics_['f_semi'] = metrics['f']
        flattened_metrics = {}
        for outer_key, outer_value in metrics_.items():
            for inner_key, inner_value in outer_value.items():
                for class_name, class_metrics in inner_value.items():
                    for metric_name, metric_value in class_metrics.items():
                        wandb_key = f"{outer_key}/{inner_key}/{class_name}/{metric_name}"
                        flattened_metrics[wandb_key] = metric_value
        if self.wandb_log is not None:
            self.wandb_log.log(flattened_metrics)
        

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