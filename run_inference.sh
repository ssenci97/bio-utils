#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ESM3
#SBATCH --output=logs/esm3-%A.log

set -euo pipefail

TOKEN="$1"

source ~/scratch/miniconda/bin/activate ~/scratch/miniconda/envs/dgx_esm3ppi
which python

# Clone repository if missing and strip .git folder to prevent submodule conflicts
if [ ! -d "src/ESM3-PPISites" ]; then
    git clone --depth 1 https://github.com/RitAreaSciencePark/ESM3-PPISites src/ESM3-PPISites
    rm -rf src/ESM3-PPISites/.git
fi

mkdir -p data/inference

echo "[SLURM-INFO] Starting ESM3 inference at: $(date)"
python3 src/ESM3-PPISites/run_prediction.py \
    --input data/contacts/contacts_sequences_inter.csv \
    --output data/inference/PDB_sequence_inference_results.csv \
    --token "$TOKEN"