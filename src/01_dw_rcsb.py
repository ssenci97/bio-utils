#!/usr/bin/env python3
"""Download mmCIF files from RCSB PDB using a list of PDB IDs or an input file."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import os
import re
import resource
import sys
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://files.rcsb.org/download/{}.cif"
DEFAULT_JSON_PATH = "configs/param_configs.json"
DEFAULT_OUT_DIR = "data/cifs"
ID_REGEX = re.compile(r"^[a-zA-Z0-9]{4}$")

###################################### ARGUMENT PARSING AND VALIDATION
def parse_args():
    parser = argparse.ArgumentParser(description="Download mmCIF files from RCSB PDB.")
    parser.add_argument("-i", "--input", type=str, help="Comma-separated PDB IDs or path to file.")
    parser.add_argument("-o", "--output", "--outdir", type=str, default=DEFAULT_OUT_DIR, help="Output directory path.")
    parser.add_argument("-j", "--json", type=str, nargs="?", const=DEFAULT_JSON_PATH, default=DEFAULT_JSON_PATH, help="Path to JSON config file.")
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args()

def get_valid_ids(input_source):
    raw_tokens = []
    if os.path.isfile(input_source):
        with open(input_source, "r") as f:
            raw_tokens = re.split(r"[,\s]+", f.read())
    else:
        raw_tokens = input_source.split(",")
    valid_ids = []
    for token in raw_tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        if ID_REGEX.match(cleaned):
            valid_ids.append(cleaned.upper())
        else:
            print(f"[PYTHON-INFO] Skipping invalid 4-character PDB ID: '{cleaned}'")
    return list(dict.fromkeys(valid_ids))

###################################### MAIN EXECUTION
def main():
    start_time = time.time()
    args = parse_args()
    config = {"cif_outdir": args.output}
    if args.no_json:
        config["input"] = args.input
    else:
        config["input"] = args.input
    if args.json is not None and not args.no_json:
        json_path = Path(args.json)
        if json_path.exists():
            print(f"[PYTHON-INFO] Reading configuration from existing JSON: {json_path}")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if "cif_outdir" not in json_data:
                print(f"[PYTHON-INFO] Updating JSON configuration with missing keys: {json_path}")
                json_data["cif_outdir"] = config["cif_outdir"]
                json_data.update({k: v for k, v in config.items() if k != "input" and k not in json_data and v is not None})
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
            if not config.get("input") and "input" in json_data:
                config["input"] = json_data["input"]
            if (not config.get("cif_outdir") or config["cif_outdir"] == DEFAULT_OUT_DIR) and "cif_outdir" in json_data:
                config["cif_outdir"] = json_data["cif_outdir"]
        else:
            print(f"[PYTHON-INFO] Writing configuration to JSON: {json_path}")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump({k: v for k, v in config.items() if k != "input" and v is not None}, f, indent=2)
    outdir = Path(config.get("cif_outdir", DEFAULT_OUT_DIR))
    outdir.mkdir(parents=True, exist_ok=True)
    input_str = config.get("input")
    if not input_str:
        print("[PYTHON-INFO] Error: No input provided via command line or JSON config.")
        sys.exit(1)
    pdb_ids = get_valid_ids(str(input_str))
    print(f"[PYTHON-INFO] Process initiated for {len(pdb_ids)} unique valid IDs.")
    downloaded, skipped, failed = 0, 0, 0
    for pdb_id in tqdm(pdb_ids, desc="Downloading mmCIFs"):
        target_path = outdir / f"{pdb_id}.cif"
        if target_path.exists():
            skipped += 1
            continue
        download_url = BASE_URL.format(pdb_id)
        try:
            urllib.request.urlretrieve(download_url, target_path)
            downloaded += 1
        except Exception as err:
            failed += 1
            print(f"[PYTHON-INFO] Failed to download {pdb_id}: {err}")
    exec_time = time.time() - start_time
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] Processing complete | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    print(f"[PYTHON-INFO] Completed on {current_date} | Execution time: {exec_time:.2f}s | Max Memory: {mem_mb:.2f} MB")

if __name__ == "__main__":
    main()


