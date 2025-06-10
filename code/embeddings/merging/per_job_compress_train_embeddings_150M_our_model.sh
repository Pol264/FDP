#!/bin/bash
#SBATCH --job-name=150M_join_job
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/join_job_%A_%a.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/join_job_%A_%a.err
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=1-00:00:00
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal


module load cuda/12.1
module load python/3.10.8--gcc--8.5.0


INPUT_DIR="/leonardo_scratch/large/userexternal/pjardiya/embeddings/embeddings_150M_our_model/"
OUTPUT_DIR="/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/jobs_embedding_150M_our_model"
mkdir -p "$OUTPUT_DIR"

python3 per_job_compress_train_embeddings.py \
  --job-id "$SLURM_ARRAY_TASK_ID" \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT_DIR/job${SLURM_ARRAY_TASK_ID}_embedding_all_150M_our_model.npy" \
  --workers 32

