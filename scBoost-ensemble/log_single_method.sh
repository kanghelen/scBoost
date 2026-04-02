## Helen Kang
## Script to run single-method XGBoost baseline for each method
## 260314

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=$SCRIPT_DIR

#######################################################################
## Classic batch correction embeddings — each method individually (7)
#######################################################################
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
INPUT=/data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad
OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/single_method
mkdir -p $OUT_DIR
MAX_JOBS=6
job_count=0

echo "=== Classic embeddings: ${#METHODS[@]} individual methods ==="
for method in "${METHODS[@]}"; do
    echo "[$((job_count+1))/${#METHODS[@]}] $method"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT \
        --sample_name "SketchProcessed_classic" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
        wait
    fi
done
wait
echo "All $job_count classic single-method runs complete."


#######################################################################
## scFM embeddings — each method individually (3)
#######################################################################
METHODS_FM=(STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d)
INPUT_FM=/data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad
job_count=0

echo ""
echo "=== scFM embeddings: ${#METHODS_FM[@]} individual methods ==="
for method in "${METHODS_FM[@]}"; do
    echo "[$((job_count+1))/${#METHODS_FM[@]}] $method"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
        wait
    fi
done
wait
echo "All $job_count scFM single-method runs complete."


#######################################################################
## scFM full embeddings — each method individually (3)
#######################################################################
METHODS_FM=(STATE600M TranscriptFormer_Sapiens scGPT)
INPUT_FM=/data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad
job_count=0

echo ""
echo "=== scFM full embeddings: ${#METHODS_FM[@]} individual methods ==="
for method in "${METHODS_FM[@]}"; do
    echo "[$((job_count+1))/${#METHODS_FM[@]}] $method"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_full_embeddings" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
        wait
    fi
done
wait
echo "All $job_count scFM single-method runs complete."

#######################################################################
## Holdout splits — Classic embeddings (7 methods)
#######################################################################
echo ""
echo "=== Classic embeddings: single methods with donor holdout ==="
job_count=0
for method in "${METHODS[@]}"; do
    echo "[$((job_count+1))/${#METHODS[@]}] $method (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT \
        --sample_name "SketchProcessed_classic_holdout" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
done
wait
echo "All $job_count classic holdout single-method runs complete."


#######################################################################
## Holdout splits — scFM embeddings (3 methods)
#######################################################################
echo ""
echo "=== scFM embeddings: single methods with donor holdout ==="
job_count=0
for method in "${METHODS_FM[@]}"; do
    echo "[$((job_count+1))/${#METHODS_FM[@]}] $method (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_holdout" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
done
wait
echo "All $job_count scFM holdout single-method runs complete."


#######################################################################
## Holdout splits — scFM full embeddings (3 methods)
#######################################################################
echo ""
echo "=== scFM full embeddings: single methods with donor holdout ==="
job_count=0
METHODS_FM=(STATE600M TranscriptFormer_Sapiens scGPT)

for method in "${METHODS_FM[@]}"; do
    echo "[$((job_count+1))/${#METHODS_FM[@]}] $method (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_single_method.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_full_embeddings_holdout" \
        --method $method \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
done
wait
echo "All $job_count scFM holdout single-method runs complete."
