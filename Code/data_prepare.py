import sklearn
# import random
# from torch.utils.data import DataLoader
import os
import time
# random.seed(10)
# print(random.random()) 
import scanorama
import scanpy as sc
import argparse

import os
import tempfile

import scanpy as sc
import scvi
import seaborn as sns
import torch
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
# save_dir = tempfile.TemporaryDirectory()

import numpy as np
import pandas as pd
# import torch
# import tqdm
from PIL import Image

import subprocess

from shannonca.dimred import reduce
from shannonca.dimred import reduce_scanpy


def sca(adata, n_components, bkey="Null", args=None):
    print("Doing SCA")
    reduce_scanpy(adata, keep_scores=True,
                   keep_loadings=True, keep_all_iters=True, 
                   layer=None, key_added='sca', iters=1, n_comps=n_components)
    
def pca(adata, n_components, bkey="Null", args=None):
    print("DOING PCA")
    sc.tl.pca(adata, n_comps=n_components)
    adata.obsm["Python_PCA"] = adata.obsm["X_pca"]

def ica(adata, n_components, inplace=True, bkey="Null", args=None, **kwargs): 
    print("Doing ICA")
    from sklearn.decomposition import FastICA 
    ica_transformer = FastICA(n_components=n_components, **kwargs) 
    x_ica = ica_transformer.fit_transform(adata.X.toarray()) 
    if inplace:
        adata.obsm["X_ica"] = x_ica 
        adata.varm["ICs"] = ica_transformer.components_.T 
    else:
        return ica_transformer 

def NMF(adata, n_components, bkey="Null", args=None):
    print("DOING NMF")
    X=adata.X
    s=sklearn.decomposition.NMF(n_components=n_components)
    Y=s.fit_transform(X)
    adata.obsm['X_NMF']=Y

def Seurat(adata, n_components, bkey="Null", args=None):
    print("Doing Seurat")
    #So you need the Batch Key [value for batch]
    #And the Label Key
    start_file = args.data

    #TODO REPLACE THE FOLDER WITH TMPFILE UTILS LIBRARY

    # Call Seurat 
    copy_data = adata.copy()
    copy_data.X = adata.layers["normalized"]
    temp_folder = "tmp"
    os.makedirs(temp_folder, exist_ok=True)
    temp_file = os.path.join(temp_folder, "seurat_anddata.h5ad")
    batch_label = bkey
    start_file = "tmp/raw.h5ad"
    copy_data.write_h5ad(start_file)
    script_name = "Code/AnnData_Seurat_Pipeline.R {} {} {} {}".format(start_file, temp_file, n_components, batch_label)
    retcode = subprocess.call("Rscript --vanilla {}".format(script_name), shell=True)
    print(retcode)
    seurat_anndat = sc.read_h5ad("tmp/seurat_anddata.h5ad")
    adata.obsm["Seurat_PCA"] = seurat_anndat.obsm["X_pca"]
    adata.obsm["Seurat_CCA_Integration"] = seurat_anndat.obsm["X_integrated.cca"]
    adata.obsm["Seurat_RPCA_Integration"] = seurat_anndat.obsm["X_integrated.rpca"]
    pass


#TODO CLEAN THIS UP
def custom_method(adata, custom_func , custom_name , bkey='batch', args=None):

    print("Doing custom method {}".format(custom_name))
    ### Assuming Non InPlace modifcation
    custom_vals = custom_func(adata, bkey, args)
    #make this an if stament
    adata[custom_name] = custom_vals 

#TODO FINISH THIS for laoding customo data 
def load_custom_data(adata, data_file, name, bkey='batch', args=None):

    # load data
    data_loaded = None #TODO FINISH THIS
    adata[name] = data_loaded

def harmony(adata, n_components, bkey= 'batch', args=None):
    # (adata=adata, n_components=n_components, bkey=bkey, args=args)
    # This requires a key to cehck
    print("Doing Harmony") 
    #USING PCA SHAPE
    sc.tl.pca(adata, n_comps=n_components) # make PCA if not done
    sc.external.pp.harmony_integrate(adata, key=bkey) #TODO pass other keywords in

    adata.obsm["Harmony"] = adata.obsm["X_pca_harmony"] # Name fix

def geosketch(adata, n_components, bkey='batch', args=None):
    print("Geosketch")
    GEOSKETCH = 5000
    adata_sc = adata.copy()

    ## Please remember to normalize and apply log transform following Scanorama Nat. Protocol paper!
    # sc.pp.normalize_total(adata_sc, target_sum=1e4)
    # sc.pp.log1p(adata_sc)

    # List of adata per batch
    batch_cats = adata_sc.obs[bkey].cat.categories
    adatas = [adata_sc[adata_sc.obs[bkey] == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adatas, sketch = True,
                               sketch_method = 'geosketch',
                               sketch_max = GEOSKETCH,
                               dimred = n_components)
    
    adata_sc.obsm["Scanorama_sk1"] = np.zeros((adata_sc.shape[0], adatas[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata_sc.obsm["Scanorama_sk1"][adata_sc.obs[bkey] == b] = adatas[i].obsm["X_scanorama"]

    adata.obsm["Geosketch_sk1"] = adata_sc.obsm["Scanorama_sk1"]

def scanorama_f(adata, n_components, bkey='batch', args=None):
    print("Doing Scanorama")
    adata_sc = adata.copy()
    
    # # List of adata per batch
    batch_cats = adata_sc.obs[bkey].cat.categories
    adatas = [adata_sc[adata_sc.obs[bkey] == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adatas, sketch = False,  dimred = n_components)
    
    adata_sc.obsm["Scanorama"] = np.zeros((adata_sc.shape[0], adatas[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata_sc.obsm["Scanorama"][adata_sc.obs[bkey] == b] = adatas[i].obsm["X_scanorama"]
   
    adata.obsm["Scanorama"] = adata_sc.obsm["Scanorama"]

def do_scVI(adata, n_components, bkey='batch', args=None):
    # do scVI and then CANVI

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=bkey)
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=n_components, gene_likelihood="nb")
    model.train()

    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key=args.label_key,
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)

    SCANVI_LATENT_KEY = "X_scANVI"
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)

    pass

def do_scVI_only(adata, n_components, bkey='batch', args=None):
    # do scVI and then CANVI

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=bkey)
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=n_components, gene_likelihood="nb")
    model.train()

    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

def transcriptonomic(adata, n_components, bkey, args=None):
    copy_data = adata.copy()
    copy_data.X = adata.layers["counts"]
    temp_folder = "transcriptonomic_temp"
    os.makedirs(temp_folder, exist_ok=True)
    start_file = os.path.join(temp_folder, "transcriptomoc_start.h5ad")
    copy_data.write_h5ad(start_file)

    # call subprocess

    start_file #File to save
    end_file = "transcriptonomic_temp/Embeddings.h5ad" #File to load
    
    script_name = "DataPrepare/transcriptonmics_inference.bash {} {} {} {}"
    retcode = subprocess.call("bash {}".format(script_name), shell=True)
    print("STARTING SCRIPT")
    print(retcode)

    transcripotnomic_anndata =  sc.read_h5ad(end_file) #LOAD
    
    adata.obsm["Transcriptonomic_Embeddings"] = transcripotnomic_anndata.obsm['embeddings']
    # adata.obsm["Seurat_RPCA_Integration"] = seurat_anndat.obsm["X_integrated.rpca"]






parser = argparse.ArgumentParser(description='Singular Cell Classifications')
parser.add_argument('-data', metavar='DIR', nargs='?', default='./',
                    help='path to dataset (default: ???)')
parser.add_argument('-destination', metavar="DIR",nargs='?', default='./',
                    help='path to save resulting data (default: ???)')
parser.add_argument('-batch_key', help="Value to use as sample origin for categories")
parser.add_argument('-label_key', help="Value to use as ground truth for labels." \
" Addtionaly if splitting is on it will ensure that this category is used  to split dataset " \
"Will create a new colum with labe_key_code with number rather than names [if orignal column is number already then extra column can be ignored]")

parser.add_argument('-train_val_split', type=float, default=0.1, help="how to split train/val, default ot 10 percent")
parser.add_argument('-method', default='all', help= "which method to use, default to all of them")
parser.add_argument('-n_comp', type=int, default=100, help="Compoment count")
parser.add_argument('--split_only', action='store_true', help="No anayalsis just resplit data, data wont rewrite itself")
parser.add_argument('--no_split', action='store_true', help='dont split data into train-val sets')
parser.add_argument('--clean_names', action='store_true', help="Changes names to be more friendly to plots, default to off")
parser.add_argument('--do_normalization', action='store_true', help="Perform Normalization step. Use this if data is not normalized")

#add learnig rate + momentum stuff

def do_CA(adata, n_components, bkey="", args=None):
    sca(adata=adata, n_components=n_components,   bkey=bkey, args=args)
    # ica(adata, n_components=n_components)
    pca(adata=adata, n_components=n_components,   bkey=bkey, args=args)
    NMF(adata=adata, n_components=n_components,   bkey=bkey, args=args)

def integration_methods(adata, n_components, bkey="", args=None):
    geosketch(adata=adata, n_components=n_components,   bkey=bkey, args=args)
    harmony(adata=adata, n_components=n_components,   bkey=bkey, args=args)
    scanorama_f(adata=adata, n_components=n_components,   bkey=bkey, args=args)
    do_scVI(adata=adata, n_components=n_components,   bkey=bkey, args=args)

def do_all_noseurat(adata, n_components, bkey, args=None):
    ## Note: parallelize running the methods
    
    do_CA(adata=adata, n_components=n_components, bkey=bkey, args=args)
    integration_methods(adata=adata, n_components=n_components, bkey=bkey, args=args)

def do_all_seurat(adata, n_components, bkey, args=None):
    ## Note: parallelize running the methods

    do_CA(adata=adata, n_components=n_components, bkey=bkey, args=args)
    # integration_methods(adata=adata, n_components=n_components, bkey=bkey, args=args)
    integration_methods(adata=adata, n_components=n_components, bkey=bkey, args=args)

    Seurat(adata=adata, n_components=n_components, bkey=bkey, args=args)
    # transcriptonomic(adata=adata, n_components=n_components, bkey=bkey, args=args)






"""" Changes names to be more user friendly might violate standards, better for graphs """
default_replace_dict = {
    "X_NMF" : "NMF",
    "X_sca_1" : "SCA_1",
    "X_pca" : "Python_PCA",
    'Seurat_CCA_Integration' : "Seurat_CCA_Integration",
    'Seurat_PCA':"Seurat_PCA",
    'Seurat_RPCA_Integration':"Seurat_RPCA_Integration",
    "X_pca_harmony" : "Harmony",
    "Geosketch_sk1" : "Geoksketch" ,
    "Scanorama" : "Scanorama",
    "X_scVI" : "scVI",
    "X_scANVI" : "scANVI",
}

def clean_names(anndata, replcae_dict=default_replace_dict):
    ndata = anndata.copy()
    del ndata.obsm
    for key,value in replcae_dict.items():
        entry = anndata.obsm.get(key,None)
        if entry is not None : #if entry doesn't exist just skip it
            ndata.obsm[value]= entry

    return ndata


def split_entry(check, i, per_split, col_key):
    s = check.obs[check.obs[col_key]==i]
    index = s.index

    series = pd.Series(index)
    val_part = series.sample(frac = per_split)
    train_part = series.drop(val_part.index)
    return train_part.values, val_part.values



def split_dataset(adata, key, split_val):
    train_set = set()
    val_set = set()
    for i in adata.obs[key].unique():
        t, v = split_entry(adata, i, per_split=split_val, col_key=key)
        # print(t)
        train_set.update(t)
        val_set.update(v)
        # print(train_set)
        # break
    return adata[list(train_set)], adata[list(val_set)]

def main():
    args = parser.parse_args()

    print("Starting")

    adata = sc.read_h5ad(args.data)
    print("Raw Data Loaded")
    if args.do_normalization:
    #TODO Add the preprocessing step here??? Done
        ## Please remember to normalize and apply log transform following Scanorama Nat. Protocol paper!
        print("PreProcessing Data")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["normalized"] = adata.X.copy()
    print(adata)
    adata.X = adata.layers["normalized"]


    #TODO CHANGE TO SUPPORT LIST ENTRIESlabe
    method_dict ={
      "sca":sca,
      "pca":pca,
      "ica":ica,
      "Scanorama":scanorama_f,
      "Geosketch":geosketch,
      "Harmony":harmony,
      "Seurat":Seurat,
      "NMF": NMF,  
      "Transcriptonomics": transcriptonomic,
      "all": do_all_noseurat,
      "all_seurat": do_all_seurat,
      "scvi+scanvi": do_scVI,
      "scvi_only": do_scVI_only,
      "none": lambda x : x #placeholder method to use with split only. if this ever triggers it raseis an error
    }
    


    placholder = pd.Categorical(adata.obs[args.label_key])
    adata.obs[args.label_key + "_codes"] = placholder.codes
    # df.cc = pd.Categorical(df.cc)

    if not args.split_only:

        method = method_dict[args.method]
        method(adata, n_components=args.n_comp, bkey=args.batch_key, args=args, )

        adata.obs['classification_code']=adata.obs[args.batch_key].cat.codes

        print("Data summary")
        print(adata.obsm_keys)
        print("-----")
        print(adata)
    
    if args.clean_names:
        adata = clean_names(adata, default_replace_dict) 
        pass

    if not args.split_only:
        adata.write_h5ad(args.destination + "_All.h5ad")

    if not args.no_split:
        print("--- SPLITTING DATA SET ---- ")
        split_val = args.train_val_split
        t_addata, v_addata = split_dataset(adata, args.label_key, split_val) 
        ## It will be great to note that label_key is the metadata column that one would like equal split on. For example use "cell_type" column in adata.obs to have the same amount of train and validation cells in each cell type.



        t_addata.write_h5ad(args.destination + "_Train.h5ad")
        v_addata.write_h5ad(args.destination + "_Val.h5ad")
        print("--- SPLIT DONE --- ")




if __name__ == '__main__':
    main()
