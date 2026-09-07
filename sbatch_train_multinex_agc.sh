#!/bin/bash
#SBATCH --job-name=multinex_agc
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started on $(hostname) at $(date) ==="
echo "GPU allocation:"
nvidia-smi

cd $SLURM_SUBMIT_DIR

$PYTHON_BIN basicsr/train.py --opt Options/Multinex_AGC_LOL-v1.yaml

echo "=== Job finished at $(date) ==="
