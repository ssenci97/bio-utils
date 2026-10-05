#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ESM3
#SBATCH --output=logs/esm3-%A.log

set -euo pipefail

TOKEN="$1"

source ~/scratch/miniconda/bin/activate ~/scratch/miniconda/envs/dgx_esm3ppi
which python

echo "[SLURM-INFO] Starting ESM3 inference at: $(date)"
python3 src/ESM3-PPISites/run_prediction.py \
    --input data/contacts/contacts_sequences.csv \
    --output data/inference/PDB_sequence_inference_results.csv \
    --token "$TOKEN"

