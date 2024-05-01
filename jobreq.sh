#!/bin/bash
#SBATCH -A p31494
#SBATCH -p short
#SBATCH --job-name="IBM - Rotor Simulations"
#SBATCH -n 4
#SBATCH -t 00:02:00
#SBATCH --mem=1GB

cd $SLURM_SUBMIT_DIR
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dedalus3
mpiexec -n 4 python3 vpart_3periodic.py