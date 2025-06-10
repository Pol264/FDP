#!/bin/bash
#SBATCH --job-name=35M_merge_all
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/merge_all_35M_our_model.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/merge_all_35M_our_model.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod

module load cuda/12.1
module load python/3.10.8--gcc--8.5.0

JOB_DIR="/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/jobs_embedding_35M_our_model/"
OUTPUT="/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/35M_our_model_embedding_all_all.npy"

python3 joining_jobs_compress_train_embeddings_35M.py \
  --job-dir "$JOB_DIR" \
  --output "$OUTPUT"

