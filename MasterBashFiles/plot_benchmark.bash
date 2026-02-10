#!/bin/bash


DATA_DIR=Data/PipelineTest/BenchmarkFolder/Pairs.csv
FILE_NAME="barplot"
SAVE_DIR=Data/PipelineTest/BenchmarkFolder
TITLE="PlaceHolder"
FORMAT="png"

mkdir $SAVE_DIR -p

python3 Code/plot_benchmark.py \
    -data_path $DATA_DIR \
    -file_name $FILE_NAME \
    -save_folder $SAVE_DIR \
    -title_name $TITLE \
    -format $FORMAT