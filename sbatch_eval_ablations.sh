#!/bin/bash
#SBATCH --job-name=eval_multinex
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_eval_ablations_%j.out
#SBATCH --error=slurm_eval_ablations_%j.err

echo "=== Evaluating Multinex Ablations on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR

export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/basicsr:$PYTHONPATH

/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python evaluate_multinex_ablations.py

echo "=== Evaluation completed at $(date) ==="
