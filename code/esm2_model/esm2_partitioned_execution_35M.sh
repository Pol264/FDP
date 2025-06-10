#!/bin/sh
#SBATCH --job-name=35M_esm_job
#SBATCH --output=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/esm2_job-%A-%a.out
#SBATCH --error=/leonardo_work/EUHPC_A05_043/TFG_pjardi/outputs/esm2_job-%A-%a.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:3
#SBATCH --account=EUHPC_A05_043
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --array=1-10%1

#module purge
#module load GCCcore/12.3.0 zlib/1.2.13-GCCcore-12.3.0 slurm CUDA CMake GCC Si es necessita treure hastag
#python -c "import torch; print(torch.__version__); print(torch.version.cuda)" TO KNOW CUDA VERSION NEEDED TO LOAD

#source /home/pjardi/miniconda3/etc/profile.d/conda.sh
#conda activate dynamics_esm2


#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
module load cuda/12.1
module load python/3.10.8--gcc--8.5.0
export WANDB_API_KEY=70b773919c9a6f9239c023530b128c900b8963bb

srun python3 esm2_job_splitter_35M.py \
    --job_id $SLURM_ARRAY_TASK_ID \
    --total_jobs 10 \
    --fasta "/leonardo_work/EUHPC_A05_043/TFG_pjardi/uniref50_dataset/uniref50_without_validation_sequences.fasta.gz" \
    --epochs_per_job 3 \
    --model_path "/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/35M.pt" \
    --checkpoint_path "/leonardo_work/EUHPC_A05_043/TFG_pjardi/checkpoints/35M_checkpoints/checkpoint.pth" \
    --log_file "/leonardo_work/EUHPC_A05_043/TFG_pjardi/logs/training_log_35M_oficial.txt"

