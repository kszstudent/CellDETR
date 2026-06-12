import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import warnings
class SemiDataset(Dataset):
    def __init__(self,file_path,transforms=None, indices= None,):
        assert transforms is not None, "transforms must be provided"
        self.file_path = file_path
        self.file = None
        self.transforms = transforms
        self.indices = indices
    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            self.barcode = list(self.file.keys())
            if self.indices is None:  # 如果在初始化时没有设置indices
                self.indices = np.arange(len(self.barcode))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        self._open_file()
        group = self.file[self.barcode[self.indices[idx]]]
        if self.transforms is not None:
            img = self.transforms(group["image"][:])
        else:
            img = group["image"][:]
            warnings.warn("transform is None")
        
        annotation = group["annotation"][:]
        patch_index = self.barcode[self.indices[idx]]
        targets = self.get_cxcywh(group["bbox"][:])  # [num_obj,4]
        
        return {
            'img': img, 
            'annotation': annotation, 
            'patch_index': patch_index,
            'targets': targets
        }
    def get_cxcywh(self, bbox):
        """
        Converts bounding box coordinates to a tuple format.
        the coordinate system is [row_indx,col_indx] for the center coordinates
        """
        if bbox.shape[0] == 0:
            cxcywh_bbox = np.zeros((1,4))
        else:
            c_x = bbox[:,:,0].sum(axis=1)/(2*255)
            c_y = bbox[:,:,1].sum(axis=1)/(2*255)
            w = (bbox[:,1,0] - bbox[:,0,0])/255
            h = (bbox[:,1,1] - bbox[:,0,1])/255
            cxcywh_bbox = np.stack([c_x, c_y, w, h], axis=1)  ## this processces are confortable with the coco format
        return cxcywh_bbox
                   
