#!/bin/bash
#SBATCH --job-name=8M_merge_all
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/merge_all_our_model.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/merge_all_our_model.err
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

JOB_DIR="/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/jobs_embedding_8M_our_model/"
OUTPUT="/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/embedding_all_all_8M_our_model.npy"

python3 joining_jobs_compress_train_embeddings_8M.py \
  --job-dir "$JOB_DIR" \
  --output "$OUTPUT"

