#!/bin/bash
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --job-name=parse
#SBATCH --output=logs/rcsb-%A.log

set -euo pipefail
echo "[SLURM-INFO] Starting Preprocessing pipeline at: $(date)"
SEPARATOR=$(printf '%.0s.' {1..50})
IDS_FP="PDBIDS.txt"

echo "$SEPARATOR [DOWNLOAD]"
python3 src/01_dw_rcsb.py -i "$IDS_FP"
echo "$SEPARATOR [CIF DATA EXTRACTION]"
python3 src/02_extract_pdb_data.py
echo "$SEPARATOR [SEQUENCE ALIGNMENT]"
python3 src/03_align_pair_local.py
echo "$SEPARATOR [PDB CHAIN REINDEXING]"
python3 src/04_reindex_pdb.py
echo "$SEPARATOR [Ca-Ca CONTACTS]"
python3 src/05_compute_pdb_contacts.py
echo "$SEPARATOR [Bonus: CONTACT SUMMARIZATION]"
python3 src/06_summarize_contacts.py
echo "$SEPARATOR"
python3 src/06_summarize_contacts.py --dedup
echo "$SEPARATOR"
python3 src/06_summarize_contacts.py --dedup-max
echo "$SEPARATOR [Bonus: Plot Chains Coverage]"
python3 src/plot_coverage.py
