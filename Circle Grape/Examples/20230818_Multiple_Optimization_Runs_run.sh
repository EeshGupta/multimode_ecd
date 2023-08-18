#!/bin/bash

#SBATCH --job-name=Day8_ge_n=3
#SBATCH --clusters=amarel
#SBATCH --partition=gpu
#SBATCH --constraint="ampere|volta"
#SBATCH --exclude=gpu[017-018]
#SBATCH --time=03-00:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=slurm.%N.%j.out


# Load Software
# not needed - module load anaconda
module load cuda/11.7.1 cudnn/7.0.3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/sw/packages/cuda/11.7.1

# Load Environment
source activate /home/eag190/miniconda3/envs/sims_gpu1

# Run File 
srun python '/home/eag190/Multimode-Conditional-Displacements/hpc_runs/multimode_circle_grape/Grape on multiple modes/State Transfer/20230720/Running SImulation/Day8_ge_3.py'