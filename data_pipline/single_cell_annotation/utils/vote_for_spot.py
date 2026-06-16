import anndata as ad
import scanpy as sc
import os
import pandas as pd
import numpy as np
 
def vote_spot(cell2location_csv:pd,rctd_csv:pd,cell_keys:list,propor_thrd:float):
    if (cell2location_csv[cell_keys] >1).any().any():
        raise ValueError('cell2location are not proportion form !')
    cell2location_csv['main_type'] = cell2location_csv[cell_keys].apply(lambda row: sorted([c for c in cell_keys if row[c]>propor_thrd]),axis=1)
    rctd_csv['main_type'] = rctd_csv[cell_keys].apply(lambda row: sorted([c for c in cell_keys if row[c]>propor_thrd]),axis=1)
    cell2location_csv = cell2location_csv.sort_index()
    rctd_csv = rctd_csv.sort_index()
    vote_df = pd.DataFrame()
    vote_df['c2l_annotation'] = cell2location_csv['main_type']
    vote_df = vote_df.loc[rctd_csv.index]
    vote_df['rctd_annotation']= rctd_csv['main_type']
    vote_df['sample'] = vote_df.index.str.split('_').str[0]
    vote_df['vote_type'] = vote_df.apply(lambda row: list(set(row['c2l_annotation']).intersection(set(row['rctd_annotation']))),axis=1)
    vote_df_defined_1 = vote_df[vote_df['vote_type'].apply(lambda row:len(row)>0)]
    return vote_df, vote_df_defined_1

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def find_consensus_spot(cell2location_csv:pd,rctd_csv:pd,cell_list:list):
    common_index = cell2location_csv.index.intersection(rctd_csv.index)
    c2l_vectors = cell2location_csv.loc[common_index,cell_list].values
    rctd_vectors = rctd_csv.loc[common_index,cell_list].values
    similarities = [cosine_similarity(c2l_vectors[i], rctd_vectors[i]) for i in range(len(common_index))]
    sim_df = pd.DataFrame({'Cosine_Similarity':similarities},index=common_index,)
    return sim_df