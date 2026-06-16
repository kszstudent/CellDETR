import tacco
import anndata as ad
import os
import requests
import numpy as np
import math
import scanpy as sc
import pandas as pd
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.sparse as sp

def  rctd_annotation_sample(sample_list:list,adata_ref:ad,file_path:str,save_path:str,annotation_key:str,n_top_genes=3000,n_cores=10):
    annotation_list = []
    for s in sample_list:
        adata = ad.read_h5ad(os.path.join(file_path,s+'.h5ad'))
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,flavor='seurat_v3')
        adata = adata[:, adata.var.highly_variable]
        adata.obs = adata.obs.drop('in_tissue',axis=1)
        conda_env = "/t9k/mnt/.conda/envs/RCTD"
        annotation = tacco.tools.annotate_RCTD(adata,adata_ref,conda_env=conda_env,annotation_key=annotation_key,\
                                                counts_location='X',n_cores=n_cores)
        annotation.to_csv(os.path.join(save_path,s+'.csv'),index=True)
        print(f"sample {s}'s annotation has been saved")
        annotation_list.append(annotation)
    return annotation_list
"""RCTD funcation deconvolute multi samples"""

def rectd_annotation_multi(adata_st:ad,adata_ref:ad,save_path:str,save_name:str,annotation_key:str,cut_length:int = 5000, n_top_genes:int = 3000,n_cores:int = 10):
    times = math.ceil(len(adata_st)/cut_length)
    ann_df_list = []
    conda_env = "/t9k/mnt/.conda/envs/RCTD"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"folder '{save_path}'have been created !")
    for t in range(times):
        adata = adata_st[t*cut_length:(t+1)*cut_length,:]
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,flavor='seurat_v3')
        adata = adata[:, adata.var.highly_variable]
        annotation = tacco.tools.annotate_RCTD(adata,adata_ref,conda_env=conda_env,annotation_key=annotation_key,\
                                                counts_location='X',n_cores=n_cores)
        annotation.to_csv(os.path.join(save_path,f'part{t}-'+save_name),index=True)
        print(f"file {t}_{save_name} has been saved")
        ann_df_list.append(annotation)
    ann_df_all = pd.concat(ann_df_list,axis=0,join='outer')
    ann_df_all.to_csv(os.path.join(save_path,'all-'+save_name),index=True) 
    return ann_df_all
"""check the h5ad file's structure"""

def check_hdf5_groups(file_path):
    with h5py.File(file_path, 'r') as f:
        # 遍历文件中的所有顶层对象（组和数据集）
        def recursive_check(group, path=''):
            for key in group:
                item = group[key]
                # 打印对象路径和类型
                if isinstance(item, h5py.Group):
                    print(f"Group: {path}/{key}")
                    # 如果是组，递归查看子组
                    recursive_check(item, path=f"{path}/{key}")
                elif isinstance(item, h5py.Dataset):
                    print(f"Dataset: {path}/{key}")
                else:
                    print(f"Unknown type: {path}/{key}")
        
        # 从顶层开始递归遍历
        recursive_check(f)


def proecess_st_tec_data(adata1):
    adata = adata1.copy()
    adata.X = np.nan_to_num(adata.X)
    adata.X = sp.csr_matrix(adata.X)
    adata.X = adata.X.astype('int32')
    adata.obs.index = adata.obs['sample'].astype(object)+'_'+ adata.obs.index  #sample are type of category
    adata.obs = adata.obs.drop('in_tissue',axis=1)
    return adata
    
