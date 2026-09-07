#!/bin/bash
#SBATCH --job-name=multinex_s1
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_strat1_%j.out
#SBATCH --error=slurm_strat1_%j.err

echo "=== Training Multinex Strategy 1 (Prior Only) on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

PYTHON_BIN="/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

$PYTHON_BIN basicsr/train.py --opt Options/Multinex_Strategy1_Prior_LOL-v1.yaml

echo "=== Finished at $(date) ==="
