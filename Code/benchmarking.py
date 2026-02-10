import os
import numpy as np
np.float_ = np.float64
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
# import scvi
import time
# scvi.settings.seed = 0
import anndata
# import scanorama
# from harmony import harmonize

import argparse


def do_bench_marks(ann_data, list_to_check, batch_key, label_key, result_file, *args, **kwargs):

    #make sure these match
    obsm_keys = list_to_check# TODO CHANGE

    biocons = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=True)
    addata= ann_data
    sc_bm = Benchmarker( 
        addata,
        batch_key= batch_key,
        label_key= label_key,
        embedding_obsm_keys=obsm_keys,
        n_jobs=8,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=BatchCorrection(),
    )
    start = time.time()
    sc_bm.prepare() 
    sc_bm.benchmark()
    end = time.time()
    print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")

    data=sc_bm.get_results(min_max_scale=False)
    csv_name = result_file
    print("--- saving results ---")
    data.to_csv(csv_name)

    pass

parser = argparse.ArgumentParser(
                    prog='benchmark',
                    description='Benchamrk Anylaysis')
                    # epilog='Text at the bottom of help')

parser.add_argument('-data', metavar='DIR', nargs='?', default='./',
                    help='AnnData object data')
parser.add_argument('-destination_folder', metavar="DIR",nargs='?', default='./',
                    help='Path to store the results. Will create if nonexistant')
parser.add_argument('-destination_name',
                    help = "name of results")
parser.add_argument('-batch_key', help="Value to use as sample origin for categories")
parser.add_argument('-label_key', help="Value to use as ground truth for labels")
parser.add_argument('-methods_to_examine', help="List of Methods in the AnnData.obsm value to compare too",
                         nargs='+',)
parser.add_argument('-do_all', action='store_true')

if __name__ == "__main__":

    args = parser.parse_args()
    print("Starting")
    print("Argumetns : ")
    print(args)

    adata = sc.read_h5ad(args.data)
    print("AnnData Loaded")

    results_csv = os.path.join(args.destination_folder, args.destination_name + ".csv")
    os.makedirs(args.destination_folder, exist_ok=True) #make if not exist

    list_to_check = args.methods_to_examine
    if args.do_all :
        list_to_check = list(adata.obsm.keys())

    print("---- Doing benchmarking ----")
    do_bench_marks(ann_data=adata,
                   list_to_check=list_to_check, 
                   label_key=args.label_key,
                   batch_key=args.batch_key,  
                   result_file=results_csv)
    pass
