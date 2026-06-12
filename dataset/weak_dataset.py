import numpy as np
import warnings
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import h5py
CELL_LIST = ['T','Myeloid','Malignant','NK','Mast','Fibroblast','Epithelial','Endothelial','B','Plasma','SMC','Pericyte','Dendritic']
### version 2
class WeakDataset(Dataset):
    def __init__(self,h5_file_path, indices= None,transforms=None):
        self.file = h5py.File(h5_file_path, 'r')
        self.barcode = list(self.file.keys())
        if transforms is not None:
            self.transform = transforms
        if indices is not  None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.barcode))
        self.cell_list = self.file.attrs['cell_list']
        self.all_type_to_index = {cell_type: idx for idx, cell_type in enumerate(CELL_LIST)}
    def __len__(self): 
        return len(self.indices)
    def __getitem__(self, idx):
        group = self.file[self.barcode[self.indices[idx]]]
        if self.transform is not None:
            img = self.transform(group["image"][()])
        else:
            img = group["image"][()]
            warnings.warn("annotation_path is None")
        annotation = group["annotation"][()]
        annotation = self.expand_proportions_to_all_cells(annotation)
        patch_index = self.barcode[idx]
        targets = self.get_cxcywh(group["bbox"][()])  # [num_obj,4]
        return {'img':img, 
                'annotation':annotation, 
                'patch_index':patch_index,
                'targets':targets}
    def get_cxcywh(self,bbox):
        """
        Converts bounding box coordinates to a tuple format.
        the coordinate system is [row_indx,col_indx] for the center coordinates
        """
        if bbox.shape[0] == 0:
            cxcywh_bbox = np.zeros((1,4))
        else:
            c_y = bbox[:,:,0].sum(axis=1)/(2*255)
            c_x = bbox[:,:,1].sum(axis=1)/(2*255)
            h = (bbox[:,1,0] - bbox[:,0,0])/255
            w = (bbox[:,1,1] - bbox[:,0,1])/255
            cxcywh_bbox = np.stack([c_x, c_y, w, h], axis=1)
        return cxcywh_bbox
    def expand_proportions_to_all_cells(self,tissue_proportions):
        full_proportions = np.zeros(len(CELL_LIST))
        for cell_type, proportion in zip(self.cell_list, tissue_proportions):
            if cell_type in self.all_type_to_index:
                idx = self.all_type_to_index[cell_type]
                full_proportions[idx] = proportion
        return full_proportions

def weak_collate_fn(batch):
    imgs = torch.stack([torch.tensor(item['img']).detach().clone() for item in batch])
    annotations = torch.stack([torch.tensor(item['annotation']) for item in batch])
    patch_indices = [item['patch_index'] for item in batch]
    targets = [item['targets'] for item in batch]  # return the uneuqal list
    return {
        'img': imgs,
        'annotation': annotations,
        'patch_index': patch_indices,
        'targets': targets
    }
### coco format stacking
def coco_collate_fn(batch,data_type=None):
    assert data_type in ['weak','semi'], "data_type must be 'weak' or 'semi'"
    if isinstance(batch[0]['img'],tuple):   # the argument 
        imgs = torch.stack([item['img'][0].clone() for item in batch])
    else:
        imgs = torch.stack([item['img'].clone() for item in batch])
    if data_type == 'weak':
        targets = tuple(
            {
                'image_id': item['patch_index'],
                'boxes': torch.tensor(item['targets'],dtype=torch.float32),
                'labels': torch.zeros(len(item['targets']), dtype=torch.int64),
                'weak_label': torch.tensor(item['annotation'],dtype=torch.float32)
            }for item in batch)
    elif data_type == 'semi':
        targets = tuple(
            {
                'image_id': item['patch_index'],
                'boxes': torch.tensor(item['targets'],dtype=torch.float32),
                'labels': torch.tensor(item['annotation'],dtype=torch.float32),
            }for item in batch)
    return  imgs,targets