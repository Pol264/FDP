#!/bin/bash
#SBATCH --job-name=autoencoder_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:3
#SBATCH --time=4-00:00:00
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/train_autoencoder_%j.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/train_autoencoder_%j.err
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod


module load python/3.10.8--gcc--8.5.0

module load cuda/12.1    # Adjust CUDA version based on Leonardo's current setup


export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=32

srun python autoencoder_parallelized_35M.py

