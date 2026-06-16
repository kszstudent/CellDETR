import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import anndata as ad
import os



def quary_spot_compo(x,y,distence_thre,seg_gdf):
   """getting annotation of cellvit for every spot"""
   annotation = seg_gdf[seg_gdf.geometry.centroid.distance(Point(x,y))<distence_thre]
   return annotation['class'].value_counts(normalize=True).to_dict()


def fining_epi_can_annotation(gen_vote_df:pd,seg_path:str,adata_path:str,condition:float):
    """for fining the annotation which annotated by single cell RNA sequence"""
    gen_vote_df = gen_vote_df[(gen_vote_df['vote_type']=='Malignant') | (gen_vote_df['vote_type']=='Epithelial')]
    samples = gen_vote_df['sample'].value_counts().keys()
    obj = []
    for s in samples:
        s_df = gen_vote_df[gen_vote_df['sample']==s].copy()
        s_adata = ad.read_h5ad(os.path.join(adata_path,s+'.h5ad'))
        spot_pix_scale = round(s_adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres'])
        seg_gdf = pd.read_parquet(os.path.join(seg_path,s+'_cellvit_seg.parquet'))
        seg_gdf = gpd.GeoDataFrame(seg_gdf, geometry=gpd.GeoSeries.from_wkb(seg_gdf['geometry']))
        for i in range(len(s_df)):
            spot_index = s_df.iloc[i].name
            x,y = s_adata.obs.loc[spot_index.split('_')[1]][['pxl_row_in_fullres','pxl_col_in_fullres']]
            annotation = quary_spot_compo(x,y,spot_pix_scale,seg_gdf)
            cells = [k for k,v in annotation.items() if v>=condition]
            if  any(x in cells for x in ['Neoplastic', 'Epithelial']):
                if ((cells[0] =='Neoplastic') & (s_df.loc[spot_index,'vote_type']=='Malignant')) or (cells[0] == s_df.loc[spot_index,'vote_type']):
                    s_df.loc[spot_index,'HE_type'] = 1 # '1'denotes keeping
                else: s_df.loc[spot_index,'HE_type'] = 2  #'2 denotes converting
            else: s_df.loc[spot_index,'HE_type'] = 0   #'0'denotes removement  
        obj.append(s_df)
    return pd.concat(obj,axis=0,join='outer')