#!/bin/bash

EPOCHS=30
ARCH=transformer_encoder
ClassKey=cell_labels
BatchSize=256


TrainData=Data/MouseDataTraining/Clean26_Dataset_200Dim_Combo_Train.h5ad
ValData=Data/MouseDataTraining/Clean26_Dataset_200Dim_Combo_Val.h5ad

BasePath=MouseClassification/LearningRate2
mkdir $BasePath -p
# Put Names to check here
names=(
"Geosketch_sk1"
"Harmony"
"Python_PCA"
"Scanorama"
"Seurat_CCA_Integration"
"Seurat_PCA"
"Seurat_RPCA_Integration"
"X_NMF"
"X_scANVI"
"X_scVI"
"X_sca_1"
)

#List of Valid Jobs
GPU_List=(2 3 4 5 6 7)
GPU_LEN=${#GPU_List[@]}
GPU_Counter=0
#Set Max Jobs to run in backgorund
#If this crasehes make it smaller
MAX_JOB_LIST=6

LearningRateList=(0.01 0.001 0.0001)

for j in "${!names[@]}"; do
        for LearningRate in  "${LearningRateList[@]}"; do
                # echo $j
                # echo ${names[j]}
                # NAME1=${names[j]}
                # Path=${BasePath}/"${NAME1}"
                # echo $Path
                # echo ${BasePath}/${NAME1}

                # continue

                #Prevents infinte jobs running in background. Only go on if below Limit
                #Set limit based on your own memory limits. 
                while [ `jobs | wc -l` -ge $MAX_JOB_LIST ]
                do
                        sleep 5
                done

                GPU=${GPU_List[$GPU_Counter]}
                NAME1=${names[j]}
                (( GPU_Counter+=1 ))
                GPU_Counter=$(($GPU_Counter % $GPU_LEN))
                # NAME2=${names[j]}
                # ADDON=$NAME1-$NAME2

                DATASET="Custom ${NAME1}"
                echo $DATASET 
                # echo $ADDON
                Path=${BasePath}/${LearningRate}/"${NAME1}"
                echo $Path
                mkdir "$Path" -p

                echo "========================================================"
                echo "Doing ${NAME1}"

                bash DataPrepare_Parellel/Parell_subscript_learningRate.bash $TrainData $ValData \
                                $EPOCHS $GPU "$DATASET" \
                                $ARCH $ClassKey "$Path" $BatchSize $LearningRate &

                # python3  Code/main.py  -train_data $TrainData -val_data $ValData \
                #                 --epochs $EPOCHS \
                #                 --gpu $GPU \
                #                 --datasetType "$DATASET"\
                #                 --arch $ARCH \
                #                 --class_key $ClassKey \
                #                 --save_dir $Path | tee ${Path}/output.txt 
                echo "========================================================"


        done
done
wait
echo "ALL DONE"
exit

# mkdir $HarmonyPath -p 
# mkdir $PCAPath  -p
# mkdir $ScanPath -p
# mkdir $GeoPath -p
# mkdir $SeuratCCAPath -p
# mkdir $SeuratRPCAPath -p
# mkdir $SeuratPCAPath -p

# mkdir $Path1 -p
# mkdir $Path2 -p
# mkdir $Path3 -p
# mkdir $Path4 -p

# Path4

# catg_func = {
#         "raw":RawData,
#         "scan":ScanoramaData,
#         "PCA":PCA_Data,
#         "SCA":SCA_Data,
#         "ICA":ICA_Data,
#         "Harmony":Harmony_Data,
#         "geosketch":Geo_Data,
#         "NMF": NMF_Data,
#         "Multi": MultiDataset,
#         "PCA_Seurat":Seurat_PCA_Data,
#         "Seurat_CCA":Seurat_CCA_Data,
#         "Seurat_RPCA":Seurat_RPCA_Data,
#     }

# echo "========================================================"
# echo "Doing Scanorama"
# python3  Code/main.py  -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "scan"\
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $ScanPath | tee ${ScanPath}/output.txt 
# echo "========================================================"
# echo "Doing Harmony"

# python3  Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Harmony" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $HarmonyPath | tee ${HarmonyPath}/output.txt 
# echo "========================================================"
# echo "Doing PCA"
# python3 Code/main.py  -train_data $TrainData -val_data $ValData \
#         --epochs $EPOCHS \
#         --gpu $GPU \
#         --class_key $ClassKey \
#         --datasetType "PCA" \
#         --arch $ARCH \
#         --save_dir $PCAPath | tee ${PCAPath}/output.txt

# echo "========================================================"
# echo "Doing Geosketch"
# python3 Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "geosketch" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $GeoPath | tee ${GeoPath}/output.txt


# echo "========================================================"
# echo "Doing Seurat_PCA"
# python3 Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "PCA_Seurat" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $SeuratPCAPath | tee ${SeuratPCAPath}/output.txt
# echo "========================================================"
# echo "Doing Seurat_CCA"

# python3 Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Seurat_CCA" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $SeuratCCAPath | tee ${SeuratCCAPath}/output.txt

# echo "========================================================"
# echo "Doing Seurat_PCCA"
# python3 Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Seurat_RPCA" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $SeuratRPCAPath | tee ${SeuratRPCAPath}/output.txt



# python3  Code/main.py  -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Multi scan Harmony PCA"\
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $Path1 | tee ${Path1}/output.txt 

# python3  Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Multi scan PCA geosketch" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $Path2 | tee ${Path2}/output.txt 

# python3  Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Multi Harmony PCA geosketch" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $Path3 | tee ${Path3}/output.txt 


# python3  Code/main.py -train_data $TrainData -val_data $ValData \
#          --epochs $EPOCHS \
#          --gpu $GPU \
#          --datasetType "Multi scan Harmony geosketch" \
#          --arch $ARCH \
#          --class_key $ClassKey \
#          --save_dir $Path4 | tee ${Path4}/output.txt 
