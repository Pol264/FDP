#!/bin/sh
#SBATCH --job-name Cosine_FCD
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/FCD_cosine_8M-%j.out
#SBATCH --time=00:30:00
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/FCD_cosine_8M-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg


# shellcheck disable=SC2039
#module purge
#module load GCCcore/12.3.0
#module load zlib/1.2.13-GCCcore-12.3.0
#module load slurm
#ml Miniconda3
#ml CUDA
#ml CMake
#ml GCC
#source /home/pjardi/miniconda3/etc/profile.d/conda.sh
#conda activate dynamics_esm2


#LD_LIBRARY_PATH=/home/pjardi/miniconda3/envs/pytorch_env/lib
#export LD_LIBRARY_PATH

#/home/pjardi/miniconda3/envs/dynamics_esm2/bin/python /home/pjardi/TFG/FCD_embeddings.py
module load cuda/12.1
module load python/3.10.8--gcc--8.5.0
export WANDB_API_KEY=70b773919c9a6f9239c023530b128c900b8963bb

python3 FCD_embeddings.py
