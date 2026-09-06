#!/bin/bash
#SBATCH --job-name=multinex_comb
#SBATCH --partition=research
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_comb_%j.out
#SBATCH --error=slurm_comb_%j.err

echo "=== Training Multinex Combined (Strategy 1 + 3) on $(hostname) at $(date) ==="
nvidia-smi

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

PYTHON_BIN="/mnt/weka/ghakobyan/.conda/envs/lightendiff/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

$PYTHON_BIN basicsr/train.py -opt Options/Multinex_Strategy1_3_Combined_LOL-v1.yaml

echo "=== Finished at $(date) ==="
