#!/bin/bash

DATA_DIR=Data/PipelineTest/AllPairs.h5ad
DESTINATION_FOLDER=Data/PipelineTest/BenchmarkFolder
NAME=Pairs
BATCH_KEY=batch
LABEL_KEY=cell_labels
METHODS_LIST='all'
mkdir $DESTINATION_FOLDER -p 


python3 Code/benchmarking.py \
    -data $DATA_DIR\
    -destination_folder $DESTINATION_FOLDER \
    -destination_name $NAME \
    -batch_key $BATCH_KEY \
    -label_key $LABEL_KEY\
    -methods_to_examine $METHODS_LIST\
    -do_all