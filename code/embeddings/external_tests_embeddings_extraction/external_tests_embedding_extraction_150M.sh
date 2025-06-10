#!/bin/bash
#SBATCH --job-name=external_datasets_%j
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/external_datasets_%j.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/external_datasets_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=9000
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg


module load cuda/12.1
module load python/3.10.8--gcc--8.5.0

export WANDB_API_KEY=70b773919c9a6f9239c023530b128c900b8963bb

python external_tests_embedding_extraction_150M.py
