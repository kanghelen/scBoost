## Helen Kang
## Script to run concat+XGBoost for all N choose 2 pairs of methods for holdout splits
## 260314

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=$SCRIPT_DIR



# ######################################################################
# #  Holdout splits
# ######################################################################
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
N=${#METHODS[@]}
INPUT=/data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad
OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/concat_pairs
mkdir -p $OUT_DIR
MAX_JOBS=7
echo ""
echo "=== Classic embeddings: pairs with donor holdout ==="
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
N=${#METHODS[@]}
total=$(( N * (N - 1) / 2 ))
job_count=0
for ((i=0; i<N-1; i++)); do
  for ((j=i+1; j<N; j++)); do
    m1=${METHODS[$i]}; m2=${METHODS[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total] $combo (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT \
        --sample_name "SketchProcessed_classic_concat2_holdout_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
  done
done
wait
echo "All $job_count classic holdout pairs complete."


echo ""
echo "=== scFM embeddings: pairs with donor holdout ==="
METHODS_FM=(STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d)
N_FM=${#METHODS_FM[@]}
INPUT_FM=/data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad
total_fm=$(( N_FM * (N_FM - 1) / 2 ))
job_count=0
for ((i=0; i<N_FM-1; i++)); do
  for ((j=i+1; j<N_FM; j++)); do
    m1=${METHODS_FM[$i]}; m2=${METHODS_FM[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total_fm] $combo (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_concat2_holdout_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
  done
done
wait
echo "All $job_count scFM holdout pairs complete."
