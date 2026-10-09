#!/usr/bin/env python3
"""Generates AlphaFold Server input JSON with pairwise combinations across complexes."""

###################################### DEPENDENCIES AND CONSTANTS
import datetime
import itertools
import json
import resource
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

CFG_PATH = Path("configs/param_configs.json")
SEED = 42
CHAIN_IDS = ("A", "B")
DEFAULT_CFG = {
    "accession_ranges_path": "configs/accession_ranges.json",
    "alphafold_json_path": "data/inference/alphafold_input.json",
    "output_csv_path": "data/inference/all_sequences.csv",
    "uniprot_dir": "data/uniprot"
}

###################################### MAIN EXECUTION
def main():
    start_time = time.time()
    print("[PYTHON-INFO] Starting AlphaFold Server JSON input generation...")
    if not CFG_PATH.exists():
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CFG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=4))
        print(f"[PYTHON-INFO] Created default configuration file at {CFG_PATH}.")
        params = DEFAULT_CFG.copy()
    else:
        try:
            params = {**DEFAULT_CFG, **json.loads(CFG_PATH.read_text())}
        except Exception:
            params = DEFAULT_CFG.copy()
    csv_fp = Path(params["output_csv_path"])
    if not csv_fp.exists():
        raise FileNotFoundError(f"[PYTHON-INFO] Input sequences CSV not found at: {csv_fp}")
    af_json_fp = Path(params["alphafold_json_path"])
    df = pd.read_csv(csv_fp)
    complexes = sorted(df["complex"].unique())
    complex_pairs = list(itertools.combinations(complexes, 2))
    af_jobs = []
    for c1, c2 in tqdm(complex_pairs, desc="[PYTHON-INFO] Generating combinations"):
        df_c1 = df[df["complex"] == c1]
        df_c2 = df[df["complex"] == c2]
        for row1 in df_c1.itertuples():
            for row2 in df_c2.itertuples():
                job_entry = {
                    "name": f"{CHAIN_IDS[0]}:{row1.id}_x_{CHAIN_IDS[1]}:{row2.id}",
                    "modelSeeds": [str(SEED)],
                    "sequences": [
                        {"proteinChain": {"id": CHAIN_IDS[0], "sequence": str(row1.sequence), "count": 1}},
                        {"proteinChain": {"id": CHAIN_IDS[1], "sequence": str(row2.sequence), "count": 1}}
                    ]
                }
                af_jobs.append(job_entry)
    print(f"[PYTHON-INFO] Total AlphaFold Server combinations generated: {len(af_jobs)}")
    af_json_fp.parent.mkdir(parents=True, exist_ok=True)
    af_json_fp.write_text(json.dumps(af_jobs, indent=2))
    params["alphafold_json_path"] = str(af_json_fp)
    CFG_PATH.write_text(json.dumps(params, indent=4))
    exec_time = time.time() - start_time
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] Completed on {curr_date} | Execution Time: {exec_time:.2f}s | Max Memory: {max_mem:.2f} MB")

if __name__ == "__main__":
    main()