## Helen Kang
## Script to run concat+XGBoost for all N choose 2 pairs of methods for stratified splits
## 260314

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=$SCRIPT_DIR

#######################################################################
## Classic batch correction embeddings — all pairs (7 choose 2 = 21)
#######################################################################
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
N=${#METHODS[@]}
INPUT=/data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad
OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/concat_pairs
mkdir -p $OUT_DIR
MAX_JOBS=7
job_count=0
total=$(( N * (N - 1) / 2 ))

echo "=== Classic embeddings: $total pairs from ${METHODS[@]} ==="
for ((i=0; i<N-1; i++)); do
  for ((j=i+1; j<N; j++)); do
    m1=${METHODS[$i]}; m2=${METHODS[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total] $combo"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT \
        --sample_name "SketchProcessed_classic_concat2_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
      wait
    fi
  done
done
wait
echo "All $job_count classic pairs complete."


# #######################################################################
# ## scFM embeddings — all pairs (3 choose 2 = 3)
# #######################################################################
METHODS_FM=(STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d)
N_FM=${#METHODS_FM[@]}
INPUT_FM=/data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad
total_fm=$(( N_FM * (N_FM - 1) / 2 ))
job_count=0

echo ""
echo "=== scFM embeddings: $total_fm pairs from ${METHODS_FM[@]} ==="
for ((i=0; i<N_FM-1; i++)); do
  for ((j=i+1; j<N_FM; j++)); do
    m1=${METHODS_FM[$i]}; m2=${METHODS_FM[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total_fm] $combo"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_concat2_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
      wait
    fi
  done
done
wait
echo "All $job_count scFM pairs complete."


#######################################################################
## scFM full embeddings — all pairs (3 choose 2 = 3)
#######################################################################
METHODS_FM_FULL=(STATE600M TranscriptFormer_Sapiens scGPT)
N_FM_FULL=${#METHODS_FM_FULL[@]}
total_fm_full=$(( N_FM_FULL * (N_FM_FULL - 1) / 2 ))
job_count=0

echo ""
echo "=== scFM full embeddings: $total_fm_full pairs from ${METHODS_FM_FULL[@]} ==="
for ((i=0; i<N_FM_FULL-1; i++)); do
  for ((j=i+1; j<N_FM_FULL; j++)); do
    m1=${METHODS_FM_FULL[$i]}; m2=${METHODS_FM_FULL[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total_fm_full] $combo"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_full_concat2_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type stratified --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then
      wait
    fi
  done
done
wait
echo "All $job_count scFM full embedding pairs complete."


######################################################################
#  Holdout splits
######################################################################

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


#######################################################################
## Holdout splits — scFM full embeddings (3 choose 2 = 3)
#######################################################################
echo ""
echo "=== scFM full embeddings: pairs with donor holdout ==="
METHODS_FM_FULL=(STATE600M TranscriptFormer_Sapiens scGPT)
N_FM_FULL=${#METHODS_FM_FULL[@]}
total_fm_full=$(( N_FM_FULL * (N_FM_FULL - 1) / 2 ))
job_count=0
for ((i=0; i<N_FM_FULL-1; i++)); do
  for ((j=i+1; j<N_FM_FULL; j++)); do
    m1=${METHODS_FM_FULL[$i]}; m2=${METHODS_FM_FULL[$j]}
    combo="${m1}_${m2}"
    echo "[$((job_count+1))/$total_fm_full] $combo (holdout)"
    gpu_id=$((job_count % MAX_JOBS))

    python $PROJECT/example_concat_pairs.py \
        --input $INPUT_FM \
        --sample_name "TabulaSapiens_scFM_full_concat2_holdout_${combo}" \
        --methods $m1 $m2 \
        --out_dir $OUT_DIR \
        --split_type donor_id --test_size 0.1 --seed 42 \
        --gpu $gpu_id --no_batch_feature &

    job_count=$((job_count + 1))
    if (( job_count % MAX_JOBS == 0 )); then wait; fi
  done
done
wait
echo "All $job_count scFM full embedding holdout pairs complete."
