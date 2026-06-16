from cell2location.utils.filtering import filter_genes
import cell2location
from cell2location.models import RegressionModel
import numpy as np
import matplotlib.pyplot as plt 
import scanpy as sc
import anndata as ad
import pandas as pd
import math
"""Training model for reference by single cells"""
def make_sc_model(adata_sc,Method='3GEX',b_k='sample',l_k ='cell_major_type',train_epoch=250):
    selected = filter_genes(adata_sc, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    # filter the object
    adata_ref = adata_sc[:, selected].copy()
    adata_ref.obs['Method'] = Method
    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                            # 10X reaction / sample / batch
                            batch_key=b_k,
                            # cell type, covariate used for constructing signatures
                            labels_key=l_k,
                            # multiplicative technical effects (platform, 3' vs 5', donor effect)
                            categorical_covariate_keys=['Method']
                        )
    mod = RegressionModel(adata_ref)
    # view anndata_setup as a sanity check
    mod.view_anndata_setup()
    mod.train(max_epochs=train_epoch)
    mod.plot_history(20)
    return mod,adata_ref

'''getting posteriori inference'''

def sc_posteriori(model_sc,adata_ref):
    adata_ref = model_sc.export_posterior(adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500,})
    model_sc.plot_QC()
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']
    return inf_aver

'''Training model for spatial transcriptome '''

def st_model(adata_st,inf_aver,num_cells_per_spot,batch_key='sample',detection_alpha=20):
    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_st.var_names, inf_aver.index)
    adata_st = adata_st[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_st, batch_key=batch_key)
    mod = cell2location.models.Cell2location(
    adata_st, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=num_cells_per_spot,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection:
    detection_alpha=detection_alpha
    )
    mod.view_anndata_setup()
    mod.train(max_epochs=30000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
         )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(1000)
    plt.legend(labels=['full data training'])
    return mod

'''getting the annotaion'''
def st_posteriori(model,adata_st,poster_key='q05_cell_abundance_w_sf'):
    adata_st= model.export_posterior(
    adata_st, sample_kwargs={'num_samples': 1000, 'batch_size': model.adata.n_obs,}
    )
    adata_st.obs[adata_st.uns['mod']['factor_names']] = adata_st.obsm[poster_key]
    return adata_st


def preprocessing_sc(adata_sc:ad):
    adata_sc.var["mt"] = adata_sc.var_names.str.startswith("MT-")# mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata_sc.var["ribo"] = adata_sc.var_names.str.startswith(("RPS", "RPL"))  # ribosomal genes
    adata_sc.var["hb"] = adata_sc.var_names.str.contains("^HB[^(P)]")  # hemoglobin genes
    sc.pp.calculate_qc_metrics(adata_sc, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True)
    adata_sc = adata_sc[:, ~adata_sc.var['mt']]  # 去除线粒体基因
    adata_sc = adata_sc[:, ~adata_sc.var['ribo']]  # 去除核糖体基因
    adata_sc = adata_sc[:, ~adata_sc.var['hb']]  # 去除红细胞基因
    adata_sc = adata_sc[adata_sc.obs['pct_counts_mt'] < 20, :]
    adata_sc = adata_sc[adata_sc.obs['pct_counts_ribo'] < 20, :]
    filtered_cells = (
    (adata_sc.obs["n_genes_by_counts"] >= 200) &
    (adata_sc.obs["n_genes_by_counts"] <= 5000) &
    (adata_sc.obs["total_counts"] >= 500) &
    (adata_sc.obs["total_counts"] <= 30000))
    adata_sc = adata_sc[~adata_sc]
    upper_threshold = adata_sc.var['nonz_mean'].quantile(0.95)
    adata_sc = adata_sc[:, adata_sc.var['nonz_mean'] < upper_threshold]

def cut_off_run(df_saving_path:str,adata_st:ad,ref:pd,cut_length:int=60000,num_cells_of_spot:int=25):
    adata_st = adata_st.copy()
    times = math.ceil(len(adata_st)/cut_length)
    cells = ref.columns.to_list()
    anno_df_list = []
    for i in range(times):
        adata = adata_st[i*cut_length:(i+1)*cut_length,:]
        st_mod = st_model(adata_st=adata,inf_aver=ref,num_cells_per_spot=num_cells_of_spot)
        adata = st_posteriori(st_mod,adata)
        adata.obs.columns = adata.obs.columns.str.rstrip()
        annotation_df = adata.obs[cells]
        annotation_df['total'] = annotation_df[cells].sum(axis=1)
        annotation_df_pct = annotation_df[cells].div(annotation_df['total'],axis=0)
        annotation_df.to_csv(f'{df_saving_path}/10X_{i}.csv',index=True)
        annotation_df_pct.to_csv(f'{df_saving_path}/10X_pct_{i}.csv',index=True)
        anno_df_list.append(annotation_df_pct)
        print(f'finish part {i} !')
    anno_all =pd.concat(anno_df_list,axis=0,join='outer')
    anno_all.to_csv(df_saving_path+'/10X_pct_all.csv',index=True)
