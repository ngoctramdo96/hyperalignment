#!/bin/bash
#SBATCH --job-name=main_prep-test-data_RHA
#SBATCH --ntasks=1
#SBATCH --output=logs/main_prep-test-data_RHA/main_prep-test-data_RHA%j.out
#SBATCH --error=logs/main_prep-test-data_RHA/main_prep-test-data_RHA%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --partition=normal
#SBATCH --mail-user=tram.do@uni-marburg.de
#SBATCH --mail-type=END
#SBATCH --chdir=/home/dotr/paper/hyperalignment_rfMRI/main/

# Fail on errors and undefined variables
set -euo pipefail

# Debugging output
echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Load Conda properly
module purge
module load miniconda

# Initialize conda
source $CONDA_ROOT/bin/activate master-thesis

# Execute Python script
python -u main_prep-test-data_RHA.py | tee "logs/main_prep-test-data_RHA/main_prep-test-data_RHA_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="