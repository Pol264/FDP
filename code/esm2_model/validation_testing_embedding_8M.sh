#!/bin/bash
#SBATCH --job-name=parallelize_job_%A_%a
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/validation_testing_embeddings_8M-%j.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/validation_testing_embeddings_8M-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal


module load cuda/12.1
module load python/3.10.8--gcc--8.5.0

export WANDB_API_KEY=70b773919c9a6f9239c023530b128c900b8963bb

python validation_testing_embedding_8M.py
