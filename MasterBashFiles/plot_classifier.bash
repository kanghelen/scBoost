#!/bin/bash


# I think this i ssupoed to point direcory"
# I think i left this unfinifsehd
# DATA_FOLDER=Expirement/OtherDanielTest1/
DATA_FOLDER1=Expirement/PCA_Combo/
DATA_FOLDER2=Expirement/SCA_Combo/
DATA_FOLDER3=Expirement/Concat_Combo/

SAVE_DIR=Data//BenchmarkFolder
TITLE1=PCA_After_50_Epochs
FILE_NAME1=PCA_ClassificationAccuracy

TITLE2=SCA_After_50_Epochs
FILE_NAME2=SCA_ClassificationAccuracy

TITLE3=Concat_After_50_Epochs
FILE_NAME3=CONCAT_ClassificationAccuracy

FORMAT="png"

mkdir $SAVE_DIR -p

python3 Code/plot_classifier.py \
    -data_folder $DATA_FOLDER1 \
    -file_name $FILE_NAME1 \
    -save_folder $SAVE_DIR \
    -title_name $TITLE1 \
    -format $FORMAT

python3 Code/plot_classifier.py \
    -data_folder $DATA_FOLDER2 \
    -file_name $FILE_NAME2 \
    -save_folder $SAVE_DIR \
    -title_name $TITLE2 \
    -format $FORMAT

python3 Code/plot_classifier.py \
    -data_folder $DATA_FOLDER3 \
    -file_name $FILE_NAME3 \
    -save_folder $SAVE_DIR \
    -title_name $TITLE3 \
    -format $FORMAT