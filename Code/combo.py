
# import scanorama




# import pandas as pd
from shannonca.dimred import reduce_scanpy

import scanpy as sc
import anndata as ad
# import scanpy.external as sce
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import argparse
from itertools import combinations



# combine the top dimensionality
def concat_combo(anndata, list_to_combine, concat_size, dim_size):

    print("Raw concat data")
    combo_lists = combinations(list_to_combine, concat_size)

    for vals_to_combine in combo_lists:

        string_val = "Raw_Concat:_" + "+".join(vals_to_combine)
        #TODO ALL THIS VALUE
        # com_data = np.hstack([anndata.obsm[name][:,0:cap] for name in vals_to_combine])

        com_data = np.hstack([anndata.obsm[name] for name in vals_to_combine])

        anndata.obsm[string_val]=com_data

    pass

# combine the dimensioanlity then perform PCA
def pca_combo(anndata, list_to_combine, concat_size, dim_size):
    print("PCA on concat data")
    combo_lists = combinations(list_to_combine, concat_size)

    for vals_to_combine in combo_lists:

        string_val = "PCACombo:_" + "+".join(vals_to_combine) 
        com_data = np.hstack([anndata.obsm[name] for name in vals_to_combine])
        pca_data = sc.pp.pca(com_data, n_comps=dim_size)

        anndata.obsm[string_val]=pca_data

    pass


#Combinen the dimensionalit then perform sca
def sca_combo(anndata, list_to_combine, concat_size, dim_size):
    print("SCA on concat data")
    combo_lists = combinations(list_to_combine, concat_size)
    #  n_adata.obsm[str_val]
    t_combo_lists = combinations(list_to_combine, concat_size)
    val_1 = t_combo_lists.__next__()
    t_data =  np.hstack([anndata.obsm[name] for name in val_1])
    t_annData = ad.AnnData(t_data)
    t_annData.X = np.zeros(t_data.shape)

    for vals_to_combine in combo_lists:

        concat_string_val = "+".join(vals_to_combine)
        string_val = "SCA_COMBO:_" + "+".join(vals_to_combine)
        com_data = np.hstack([anndata.obsm[name] for name in vals_to_combine])
        # pca_data = sc.pp.pca(com_data, n_comps=dim_size)
        t_annData.layers[concat_string_val] = com_data
        reduce_scanpy(t_annData, keep_scores=True,
            keep_loadings=True, keep_all_iters=True, 
            layer=concat_string_val, key_added='[{}]_ComboSCA'.format(concat_string_val), iters=1, n_comps=dim_size)
        anndata.obsm[string_val]=t_annData.obsm['X_[{}]_ComboSCA_1'.format(concat_string_val)]
    pass

def all_combo(anndata, list_to_combine, concat_size, dim_size):
    concat_combo(anndata, list_to_combine, concat_size, dim_size)
    pca_combo(anndata, list_to_combine, concat_size, dim_size)
    sca_combo(anndata, list_to_combine, concat_size, dim_size)

parser = argparse.ArgumentParser(description='combo methods')

parser.add_argument('-data', metavar='DIR', nargs='?', default='./',
                    help='AnnData object data')
parser.add_argument('-destiantion', metavar="DIR",nargs='?', default='./',
                    help='Store Resulting AnnData file')
parser.add_argument('-batch_key', help="Value to use as sample origin for categories")
parser.add_argument('-label_key', help="Value to use as ground truth for labels")
parser.add_argument('-method', help="what combo techiuqes to be used")
parser.add_argument('-methods_to_combine', help='methods you wanna conact, use names stored in AnnData.obsm',
                    nargs='+', )
parser.add_argument('-concat_size', type=int, default=2, help="how many methods to combien, should not exceed method list")
parser.add_argument('-n_comp', type=int, default=100, help="Compoment count")


if __name__ == "__main__":

    args = parser.parse_args()
    print("args", args)

    list_to_combine = args.methods_to_combine
    anndata = sc.read_h5ad(args.data)
    save_file = args.destiantion

    batch_key = args.batch_key
    label_key = args.label_key
    concat_size = args.concat_size 
    dim_size = args.n_comp

    method_dict = {
        "concat" : concat_combo,
        "pca" : pca_combo,
        "sca" : sca_combo,
        "all" : all_combo,
    }

    method = method_dict[args.method]

    print("Old Columns : " , anndata.obsm.keys() )

    print("combing values")
    method(anndata, list_to_combine, concat_size, dim_size)

    print("processed anndata")
    print(anndata)

    print("New Columns : " , anndata.obsm.keys() )
    anndata.write_h5ad(save_file)


    pass