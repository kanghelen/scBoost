#!/bin/bash

TrainData=$1
ValData=$2 
EPOCHS=$3
GPU=$4
DATASET=$5
ARCH=$6
ClassKey=$7
Path=$8
BatchSize=$9



# exit #TEST
LogDir="$Path"/output.txt
# echo $LogDir

echo "========================================================"
echo "Doing {$DATASET}"
echo TrainData $TrainData
echo ValData $ValData
echo EPOCHS $EPOCHS
echo GPU $GPU
echo DATASET $DATASET
echo ARCH $ARCH
echo ClassKey $ClassKey
echo Path $Path
echo BatchSize $BatchSize
python3  Code/main.py  -train_data $TrainData -val_data $ValData \
                --epochs $EPOCHS \
                --gpu $GPU \
                --datasetType "$DATASET"\
                --arch $ARCH \
                --class_key $ClassKey \
                --batch-size $BatchSize \
                --save_dir "$Path" | tee "$LogDir"
echo "========================================================"
