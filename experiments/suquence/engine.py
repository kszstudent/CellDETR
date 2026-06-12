import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
from tqdm import tqdm
import wandb
from torch.utils.data import ConcatDataset
from dataset import build_dataset, build_loader
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
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, args.optimizer.lr_drop)
        self.test_dataset = build_dataset(self.args, split='test')
        self.test_loader = build_loader(self.args, self.test_dataset,split='test')
    
    def train(self):
        train_fold1 =self.args.dataset['train'].fold
        train_dataset1 = build_dataset(self.args, split='train')
        val_dataset1   = build_dataset(self.args, split='val')
        self.args.dataset['train'].fold = self.args.dataset['val'].fold
        self.args.dataset['val'].fold = train_fold1
        train_dataset2 = build_dataset(self.args, split='train')
        val_dataset2   = build_dataset(self.args, split='val')
        print(f'train_fold1: {train_fold1}, train_fold2: {self.args.dataset["train"].fold},test_fold: {self.test_dataset.fold}')
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])
        torch.manual_seed(self.args.experiment.seed)
        np.random.seed(self.args.experiment.seed)
        total_size = len(train_dataset)
        val_size = int(0.2 * total_size)
        indices = torch.randperm(total_size).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        train_loader = build_loader(self.args, train_subset)
        val_loader = build_loader(self.args, val_subset,split='val')
        # train_dataset = build_dataset(self.args, split='train')
        # val_dataset   = build_dataset(self.args, split='val')
        # train_loader = build_loader(self.args, train_dataset)
        # val_loader = build_loader(self.args, val_dataset,split='val')
        for epoch in range(self.args.optimizer.epochs):
            self.model.train()
            self.one_epoch_train(train_loader,epoch)
            self.one_epoch_eval(val_loader,epoch)
            self.test()
            self.lr_scheduler.step()
            if epoch>self.args.optimizer.start_saving_epoch and epoch%self.args.optimizer.save_per_epoch == 0:
                self.saving_model(epoch)
            
    def one_epoch_train(self,train_loader,epoch):
        train_tqdm = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{self.args.optimizer.epochs}')
        for img,targets in train_tqdm:
            img = img.tensors.to(self.device)
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
    @torch.no_grad()
    def one_epoch_eval(self,eval_loader,epoch):
        self.model.eval()
        eval_tqdm = tqdm(eval_loader, desc=f'eval Epoch {epoch+1}/{self.args.optimizer.epochs}')
        for img,targets in eval_tqdm:
            img = img.tensors.to(self.device)
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
        model = self.model.to('cuda')
        for imgs,targets in tqdm(self.test_loader):
            imgs = imgs.tensors.to('cuda')
            model.eval()
            outputs_ = {}
            with torch.no_grad():
                outputs = model(imgs)
                bf_logist = outputs['pred_logits'].sigmoid()
                class_logits = outputs['decouple_class_logits']
                class_logits[torch.where(bf_logist<0.4)[:2]] = -100
            outputs_['pred_logits'] = class_logits.clone().cpu()
            outputs_['pred_boxes'] = outputs['pred_boxes'].clone().cpu()
            orig_target_sizes = torch.stack([torch.tensor(t["boxes"].canvas_size) for t in targets], dim=0)
            predictions = self.postprocessor['bbox'](outputs_, orig_target_sizes)
            for p in predictions:
                # convert boxes
                p['boxes'] = box_ops.box_xyxy_to_cxcywh(p['boxes'])

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
