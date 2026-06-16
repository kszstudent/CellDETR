import anndata as ad
import mygene
import pandas as pd
def convert_genid_name(adata,save_path=None):
    genes_cv = [g for g in adata.var.index.to_list() if g.startswith('ENSG')]
    print(f'there have {len(genes_cv)} genes needding to be converted!')
    mg = mygene.MyGeneInfo()
    genes_df = mg.querymany(genes_cv, scopes='ensembl.gene', species='human',as_dataframe=True)
    if save_path != None:
        genes_df.to_csv(save_path,index=True)
    genes_remove = genes_df.index[genes_df.index.duplicated()].tolist()
    genes_remove = genes_remove+genes_df[genes_df['notfound']==True].index.to_list() ### remove the duplicated and unfound gene-IDs
    genes_df = genes_df[~genes_df.index.isin(genes_remove)]
    map_dict = pd.Series(genes_df.symbol.values,index=genes_df.index).to_dict()
    adata.var = adata.var.rename(index=map_dict)
    print(f'there has {len(adata.var.index[adata.var.index.duplicated()].to_list())} needding to be integrated !')
    df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names).fillna(0)
    df = df.groupby(df.columns,axis=1).sum()
    new_adata = ad.AnnData(X=df.values)
    new_adata.obs = adata.obs
    new_adata.var.index = df.columns
    return new_adata