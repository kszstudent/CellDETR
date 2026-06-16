%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import anndata as ad
import os
import scanpy as sc
import numpy as np
from utils import RCTD
import pandas as pd
import scipy.sparse as sp
import hdf5plugin
## args
ref_path = str(input('Enter reference path:'))
st_path = str(input('Enter st data path:'))
save_path = 
cut_off_length = 5000
## run 
adata_ref = ad.read_h5ad(ref_path)
adata_ref.X = adata_ref.X.astype('float32')
st_adata = sc.read_h5ad(st_path)
st_adata.obs.index = st_adata.obs['sample'].astype(object)+'_'+st_adata.obs.index
annotation = RCTD.rectd_annotation_multi(adata_st=st_adata,adata_ref=adata_ref,save_path='./annotation_output/Prostate/RCTD',\
                                         save_name='10x_Prostate_1.csv',cut_length=5000,annotation_key='major_cell_type',n_top_genes=5000,n_cores=50)