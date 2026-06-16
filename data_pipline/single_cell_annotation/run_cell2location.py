import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)
from utils import c2l_annotation
import anndata as ad
import pandas as pd
import scipy
import numpy as np
import hdf5plugin
import scanpy as sc
import os
import json
import sys
import tqdm



with open('./cell2location_arg.json', 'r') as f:
    params = json.load(f)
model_input = params['model_input']
reference_path = '/t9k/mnt/zsk/workspace/HEST/project/Data/dataset/sc'
st_adata_path = './temporary_data/'
if model_input == 'sc':
    adata_sc = ad.read_h5ad(os.path.join(reference_path,params['sc_re_file']))
    adata_sc.X = adata_sc.X.astype('int32')
    sc_mod,adata_ref = c2l_annotation.make_sc_model(adata_sc,l_k='major_cell_type',)
    sc_mod.save(params['model_ref_path'],overwrite=True)
    adata_ref = c2l_annotation.sc_posteriori(sc_mod,adata_ref)
    adata_ref.to_csv(os.path.join(params['model_ref_path'],'ref.csv'))
    print('finished the training process of sc reference model')
if model_input == 'st':
    cut_off = params['cut_off']
    saving_path = params['saving_path']
    adata_st = ad.read_h5ad(os.path.join(st_adata_path,params['st_file']))
    ref = pd.read_csv(params['ref_path'],index_col=0)
    if cut_off :
        c2l_annotation.cut_off_run(saving_path,adata_st,ref,40000)
    else:
        st_mod = c2l_annotation.st_model(adata_st=adata_st,inf_aver=ref,num_cells_per_spot=25)
        adata_st = c2l_annotation.st_posteriori(st_mod,adata_st)
        cells = ref.columns.to_list()
        annotation_df = adata_st.obs[cells]
        annotation_df['total'] = annotation_df[cells].sum(axis=1)
        annotation_df_pct = annotation_df[cells].div(annotation_df['total'],axis=0)
        annotation_df_pct.to_csv(saving_path+'/10X_pct.csv',index=True)
        annotation_df.to_csv(saving_path+'/10X.csv',index=True)
