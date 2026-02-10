#!/bin/bash

DATA_DIR=Data/AllData.h5ad
# DO NO TADD .h5ad Name to desitnation
DESTINATION=Data/PipelineTest/AllRawSeuratNames
BATCH_KEY=batch
LABEL_KEY=cell_types

TRAIN_VAL_SPLIT=0.1

METHOD=all_seurat
N_COMP=100

# http://localhost:87643
# obsm: 'Scanorama', 'Scanorama_sk1',
#  'Unintegrated', 'X_pca', 'X_sca_1', 'Python_PCA',
#  'X_NMF', 'Geosketch_sk1', 'X_pca_harmony', 'Harmony',
#  'X_scVI', 'X_scANVI'


    # obsm: 'Scanorama', 'Scanorama_sk1',
    # 'Unintegrated', 'X_pca', 'X_sca_1', 'Python_PCA',
    # 'X_NMF', 'Geosketch_sk1', 'X_pca_harmony',
    # 'Harmony', 'X_scVI', 'X_scANVI', 'Seurat_PCA',
    # 'Seurat_CCA_Integration', 'Seurat_RPCA_Integration'

    # 

python3 Code/data_prepare.py -data $DATA_DIR \
                        -destination $DESTINATION \
                        -batch_key $BATCH_KEY \
                        -label_key $LABEL_KEY \
                        -train_val_split $TRAIN_VAL_SPLIT \
                        -method $METHOD \
                        -n_comp $N_COMP \
                        -no_split
                        
 