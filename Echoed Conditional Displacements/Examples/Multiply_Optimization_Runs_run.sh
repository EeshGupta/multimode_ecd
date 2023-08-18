#!/bin/bash

#SBATCH --job-name=Day7_fock_swap_5
#SBATCH --clusters=amarel
#SBATCH --partition=gpu
##### --constraint=ampere
#SBATCH --nodelist=gpu020
#SBATCH --time=02-00:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=slurm.%N.%j.out


# Load Software
# not needed - module load anaconda
module load cuda/11.7.1 cudnn/7.0.3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/sw/packages/cuda/11.7.1

# Load Environment
source activate /home/eag190/miniconda3/envs/sims_gpu1

# Run File (Enter path of code; current one is not correct)
srun python '/home/eag190/Multimode-Conditional-Displacements/hpc_runs/two_mode_ecd/State Transfer/06152023/Day7fock_swap_5.py'