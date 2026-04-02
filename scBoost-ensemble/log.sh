## Helen Kang
## Script to call example_combined_calls.py #bash
## 260313


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=$SCRIPT_DIR

# ## Stratified split (default)
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad \
#     --sample_name SketchProcessed_All_100Comp_All \
#     --methods Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type stratified --test_size 0.1 --seed 42 --gpu 0 --no_batch_feature &

# ## Tabula Sapiens with classic batch correction method embeddings and holdout one donor
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad \
#     --sample_name SketchProcessed_All_100Comp_All \
#     --methods Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type donor_id --test_size 0.1 --seed 42 --gpu 1 --no_batch_feature

# ## Tabula Sapiens with scFM embeddings stratified split
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad \
#     --sample_name TabulaSapiens_with_scFM_embeddings \
#     --methods STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type stratified --test_size 0.1 --seed 42 --gpu 2 --no_batch_feature &

# ## Tabula Sapiens with scFM embeddings stratified split with full embeddings
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad \
#     --sample_name TabulaSapiens_with_scFM_full_embeddings \
#     --methods STATE600M TranscriptFormer_Sapiens scGPT \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type stratified --test_size 0.1 --seed 42 --gpu 3 --no_batch_feature &

# ## Tabula Sapiens with scFM embeddings holdout one donor
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad \
#     --sample_name TabulaSapiens_with_scFM_embeddings \
#     --methods STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type donor_id --test_size 0.1 --seed 42 --gpu 4 --no_batch_feature

# ## Tabula Sapiens with scFM embeddings holdout one donor
# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad \
#     --sample_name TabulaSapiens_with_scFM_full_embeddings \
#     --methods STATE600M TranscriptFormer_Sapiens scGPT \
#     --out_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs \
#     --split_type donor_id --test_size 0.1 --seed 42 --gpu 5 --no_batch_feature

# ## Test Only
# python $PROJECT/test_only.py \
#   --input /data/cb/helenk/Data/OpenProblems/TabulaSapiens/260113_sketch_with_SE_scGPT_TranscriptFormer_embeddings_withPCA100d.h5ad \
#   --model /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/260314_022220_TabulaSapiens_with_scFM_embeddings_holdout_donor_id_TSP7_stacking_ensemble.pkl \
#   --sample_name TabulaSapiens_with_scFM_embeddings \
#   --split_type donor_id --holdout_value TSP7

# python $PROJECT/test_only.py \
#   --input /data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad \
#   --model /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/260314_023001_SketchProcessed_All_100Comp_All_holdout_donor_id_TSP7_stacking_ensemble.pkl \
#   --sample_name SketchProcessed_All_100Comp_All \
#   --split_type donor_id --holdout_value TSP7



# # Permute with classic batch correction method embeddings
# # Use three different embeddings (7 choose 3 = 35 combinations)
# METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
# N=${#METHODS[@]}
# INPUT=/data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad
# OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/permutations
# mkdir -p $OUT_DIR
# MAX_JOBS=6
# job_count=0

# for ((i=0; i<N-2; i++)); do
#   for ((j=i+1; j<N-1; j++)); do
#     for ((k=j+1; k<N; k++)); do
#       m1=${METHODS[$i]}; m2=${METHODS[$j]}; m3=${METHODS[$k]}
#       combo="${m1}_${m2}_${m3}"
#       echo "[$job_count/$((N*(N-1)*(N-2)/6))] Running combination: $combo"
#       gpu_id=$((job_count % MAX_JOBS))

#       python $PROJECT/example_combined_calls.py \
#           --input $INPUT \
#           --sample_name "SketchProcessed_classic_combo3_${combo}" \
#           --methods $m1 $m2 $m3 \
#           --out_dir $OUT_DIR \
#           --split_type stratified --test_size 0.1 --seed 42 \
#           --gpu $gpu_id --no_batch_feature  &

#       job_count=$((job_count + 1))
#       if (( job_count % MAX_JOBS == 0 )); then
#         wait
#       fi
#     done
#   done
# done
# wait
# echo "All ${job_count} combinations complete."


# Permute with classic batch correction method embeddings
# Use three different embeddings (7 choose 3 = 35 combinations)
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
N=${#METHODS[@]}
INPUT=/data/cb/danieled/preprocessing_classificaiton/Scanorama_Practice/Data/SketchData/SketchProcessed_All_100Comp_All.h5ad
OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs/permutations
mkdir -p $OUT_DIR
MAX_JOBS=8
job_count=0

for ((i=0; i<N-2; i++)); do
  for ((j=i+1; j<N-1; j++)); do
    for ((k=j+1; k<N; k++)); do
      m1=${METHODS[$i]}; m2=${METHODS[$j]}; m3=${METHODS[$k]}
      combo="${m1}_${m2}_${m3}"
      echo "[$job_count/$((N*(N-1)*(N-2)/6))] Running combination: $combo"
      gpu_id=$((job_count % MAX_JOBS))

      python $PROJECT/example_combined_calls.py \
          --input $INPUT \
          --sample_name "SketchProcessed_classic_combo3_${combo}" \
          --methods $m1 $m2 $m3 \
          --out_dir $OUT_DIR \
          --split_type donor_id --test_size 0.1 --seed 42 \
          --gpu $gpu_id --no_batch_feature  &

      job_count=$((job_count + 1))
      if (( job_count % MAX_JOBS == 0 )); then
        wait
      fi
    done
  done
done
wait
echo "All ${job_count} combinations complete."

## GTEX

# python $PROJECT/example_combined_calls.py \
#     --input /data/cb/scratch/ekin/vcc/scbale_data/data/gtex_v9/gtex_100k_sketch.h5ad \
#     --sample_name gtex_100k_sketch \
#     --methods STATE600M_PCA100d TranscriptFormer_Sapiens_PCA100d scGPT_PCA100d \
#     --split_type stratified --test_size 0.1 --seed 42


#######################################################################
## Extract summary metrics from all classification reports
#######################################################################
python $PROJECT/extract_summary_metrics.py \
    --outputs_dir /data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs


## Plots
OUT_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs
FIG_DIR=/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/figures
mkdir -p $FIG_DIR
python $PROJECT/plots.py \
    --input $OUT_DIR/summary_metrics_slim.tsv \
    --out   $FIG_DIR/stacked_ensemble_dot_plot.svg \
    --dpi   300


python $PROJECT/plot_barplot.py \
    --input $OUT_DIR/summary_metrics_slim.tsv \
    --out   $FIG_DIR/stacked_ensemble_barplot.svg \
    --dpi   300