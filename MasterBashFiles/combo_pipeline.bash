#!/bin/bash
DATA_DIR=Data/PipelineTest/All_Processed_Data.h5ad
SAVE_DIR=Data/PipelineTest/AllPairs.h5ad

BATCH_KEY=batch
LABEL_KEY=cell_labels
METHOD=all
METHODS_TO_COMBINE="Geosketch_sk1 Python_PCA Scanorama X_pca_harmony Seurat_CCA_Integration"
CONCAT_SIZE=2
N_COMP=100


python3 Code/combo.py    -data $DATA_DIR \
                    -destiantion $SAVE_DIR \
                    -batch_key $BATCH_KEY \
                    -label_key $LABEL_KEY \
                    -method $METHOD \
                    -methods_to_combine $METHODS_TO_COMBINE \
                    -concat_size $CONCAT_SIZE \
                    -n_comp $N_COMP \