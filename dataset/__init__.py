from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, SequentialSampler, DistributedSampler

from .pannuke import build_pannuke_dataset
from .consep import build_consep_dataset
from .monuseg import build_monuseg_dataset
from .dataset import build_cell_dataset
from util.misc import nested_tensor_from_tensor_list

def build_dataset(cfg, split='train'):
    if cfg.dataset[split].name == 'pannuke':
        dataset = build_pannuke_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'consep':
        dataset = build_consep_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'monuseg':
        dataset = build_monuseg_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'cell':
        dataset = build_cell_dataset(cfg, split=split)
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset[split].name}')
    
    return dataset

#collate_fn = lambda batch : tuple(zip(*batch))
def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return batch

def build_loader(cfg, dataset, split='train', collate_fn=collate_fn):
    _loader_cfg = cfg.loader[split]
    
    # create sampler
    sampler = None
    if cfg.distributed:
        if split in ['train','val','infer']:
            sampler = DistributedSampler(dataset, 
                                     shuffle=_loader_cfg.shuffle,
                                     num_replicas=cfg.world_size,
                                     rank=cfg.rank)
        else:
            from .loader import DistributedSamplerNoDuplicate
            sampler = DistributedSamplerNoDuplicate(dataset, 
                                     shuffle=_loader_cfg.shuffle,
                                     num_replicas=cfg.world_size,
                                     rank=cfg.rank)
    else:
        sampler = RandomSampler(dataset) if _loader_cfg.shuffle else SequentialSampler(dataset)
    # create data loader
    loader = DataLoader(dataset, sampler=sampler,
                        batch_size=_loader_cfg.batch_size,
                        num_workers=_loader_cfg.num_workers,
                        drop_last=_loader_cfg.drop_last,
                        collate_fn=collate_fn, pin_memory=split=="infer")
    # pin memory only for inference as if done in train or val, converts tv_tensors to standard torch tensors.
    return loader

### add weak dataset loader
from .weak_dataset import WeakDataset,coco_collate_fn
from .semi_dataset import SemiDataset
import h5py
import numpy as np
import os
from functools import partial
from .transforms import build_weak_semi_transforms
def build_semi_weak_dataloader(cfg):
    train_dataset = []
    val_dataset = []
    for tissue in cfg.dataset.tissue:
        file_name = f'{tissue}.h5'
        file_path =  os.path.join(cfg.dataset.root,file_name)
        with h5py.File(file_path, 'r') as f:
            data_length = len(f)
        all_indices = np.arange(data_length)
        if cfg.dataset.shuffle:
            np.random.shuffle(all_indices)
        train_indices = all_indices[:int(cfg.dataset.train_pct*data_length)]
        eval_indices = all_indices[int(cfg.dataset.train_pct*data_length):]
        if (cfg.dataset.name == 'weak') or (cfg.dataset.name == 'contrastive'):
            d_t = WeakDataset(file_path, transforms=build_weak_semi_transforms(cfg,is_train=True), indices=train_indices)
            train_dataset.append(d_t)
            d_v = WeakDataset(file_path, transforms=build_weak_semi_transforms(cfg,is_train=False), indices=eval_indices,)
            val_dataset.append(d_v)
            cc_fn = partial(coco_collate_fn,data_type='weak')
        if cfg.dataset.name == 'semi':
            d_t = SemiDataset(file_path, transforms=build_weak_semi_transforms(cfg,is_train=True), indices=train_indices)
            train_dataset.append(d_t)
            d_v = SemiDataset(file_path, transforms=build_weak_semi_transforms(cfg,is_train=False), indices=eval_indices,)
            val_dataset.append(d_v)
            cc_fn = partial(coco_collate_fn,data_type='semi')
    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.loader.train.batch_size,collate_fn=cc_fn, shuffle=cfg.loader.train.shuffle,num_workers=cfg.loader.train.num_workers,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.loader.val.batch_size,collate_fn=cc_fn, shuffle=cfg.loader.val.shuffle,num_workers=cfg.loader.val.num_workers,pin_memory=True)
    return train_loader, val_loader