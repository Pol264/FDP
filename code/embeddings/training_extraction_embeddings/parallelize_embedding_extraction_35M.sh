#!/bin/bash
#SBATCH --job-name=35M_parallelize_job_%A_%a
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/parallelize_job_%A_%a.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/parallelize_job_%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24000
#SBATCH --gres=gpu:3
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --array=0-9  # Launch 10 jobs (adjust based on N_JOBS)

module load cuda/12.1
module load python/3.10.8--gcc--8.5.0

export WANDB_API_KEY=70b773919c9a6f9239c023530b128c900b8963bb

DATA_PATH="/leonardo_work/EUHPC_A05_043/TFG_pjardi/uniref50_dataset/uniref50_without_validation_sequences.fasta.gz"
OUTPUT_DIR="/leonardo_scratch/large/userexternal/pjardiya/embeddings/embeddings_35M_our_model"
TOTAL_SEQS=67791787  # total sequences in dataset
N_JOBS=10  # Adjust total number of parallel jobs
JOB_IDX=${SLURM_ARRAY_TASK_ID}

python parallelize_embedding_extraction_pol_35M.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --total_seqs $TOTAL_SEQS \
    --job_idx $JOB_IDX \
    --n_jobs $N_JOBS

