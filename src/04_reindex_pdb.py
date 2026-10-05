#!/usr/bin/env python
"""Batch reindex PDB chain residue sequence numbers with detailed residue logging."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import io
import json
import resource
import time
import warnings
from pathlib import Path

from Bio.PDB import Chain, MMCIFParser, Model, PDBIO, PDBParser, Structure
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.Residue import Residue
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
DEFAULT_ALN_OUTDIR = "data/aln"
DEFAULT_CHAIN_OUTDIR = "data/chains"
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"

###################################### CONFIG
def resolve_config(json_arg, defaults):
    if json_arg is None:
        return defaults
    p = Path(json_arg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(defaults, indent=2))
        return defaults
    try:
        cfg = json.loads(p.read_text())
    except Exception:
        cfg = {}
    updated = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated:
        p.write_text(json.dumps(cfg, indent=2))
    return cfg

###################################### REINDEXING LOGIC
def load_mapping_table(table_path):
    df = pd.read_csv(table_path, sep="\t" if str(table_path).endswith(".tsv") else ",")
    mapping = {}
    alt_col = next((c for c in ["posalt", "old_resnum", "query_pos", "alt"] if c in df.columns), df.columns[0])
    ref_col = next((c for c in ["posref", "new_resnum", "ref_pos", "ref"] if c in df.columns), df.columns[1])
    for _, row in df.iterrows():
        try:
            alt_val = int(row[alt_col])
            ref_val = int(row[ref_col])
            mapping[alt_val] = ref_val
        except (ValueError, TypeError):
            continue
    return mapping

def reindex_pdb(pdb_path, mapping):
    parser = PDBParser(QUIET=True) if str(pdb_path).endswith(".pdb") else MMCIFParser(QUIET=True)
    struct = parser.get_structure("chain", str(pdb_path))
    chains = list(struct.get_chains())
    if not chains:
        return "", {}
    chain = chains[0]
    all_residues = list(chain)
    total_residues = len(all_residues)
    seq_pos = 0
    sequential_map = {}
    protein_residues = 0
    non_protein_removed = 0
    for residue in all_residues:
        resname = residue.get_resname().strip().upper()
        if residue.id[0] == " " and resname in protein_letters_3to1_extended:
            seq_pos += 1
            sequential_map[residue.id] = seq_pos
            protein_residues += 1
        else:
            non_protein_removed += 1
    new_struct = Structure.Structure("reindexed")
    new_model = Model.Model(0)
    new_chain = Chain.Chain(chain.id)
    new_model.add(new_chain)
    new_struct.add(new_model)
    reindexed_count = 0
    unmapped_removed = 0
    added_positions = set()
    for residue in all_residues:
        res_id = residue.id
        resname = residue.get_resname().strip().upper()
        if res_id[0] != " " or resname not in protein_letters_3to1_extended:
            continue
        qpos = sequential_map.get(res_id, res_id[1])
        target_pos = mapping.get(qpos, mapping.get(res_id[1]))
        if target_pos is not None:
            target_pos = int(target_pos)
            if target_pos in added_positions:
                unmapped_removed += 1
                continue
            res_copy = Residue((" ", target_pos, " "), residue.get_resname(), residue.get_segid())
            for atom in residue:
                res_copy.add(atom.copy())
            added_positions.add(target_pos)
            new_chain.add(res_copy)
            reindexed_count += 1
        else:
            unmapped_removed += 1
    writer = PDBIO()
    writer.set_structure(new_struct)
    buf = io.StringIO()
    writer.save(buf)
    stats = {
        "total_residues": total_residues,
        "protein_residues": protein_residues,
        "non_protein_removed": non_protein_removed,
        "reindexed_residues": reindexed_count,
        "unmapped_removed": unmapped_removed,
    }
    return buf.getvalue(), stats

def find_mapping_file(stem, aln_dir):
    candidates = [
        aln_dir / f"{stem}_positions.tsv",
        aln_dir / f"{stem}_positions.csv",
        aln_dir / f"{stem}.tsv",
        aln_dir / f"{stem}.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None

def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Batch reindex PDB chains with detailed residue statistics")
    parser.add_argument("-i", "--input", help="PDB file path or directory (defaults to chain_outdir in config)")
    parser.add_argument("-t", "--table", help="Specific mapping table file (optional for batch mode)")
    parser.add_argument("-o", "--output", help="Specific output file or directory")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON existence and update checks")
    args = parser.parse_args()
    defaults = {
        "chain_outdir": DEFAULT_CHAIN_OUTDIR,
        "aln_outdir": DEFAULT_ALN_OUTDIR,
        "clean_outdir": DEFAULT_CLEAN_OUTDIR,
    }
    cfg = resolve_config(args.json, defaults)
    chain_dir = Path(args.input) if args.input else Path(cfg["chain_outdir"])
    aln_dir = Path(cfg["aln_outdir"])
    clean_dir = Path(cfg["clean_outdir"])
    if args.output and not Path(args.output).suffix:
        clean_dir = Path(args.output)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if chain_dir.is_file():
        pdb_files = [chain_dir]
    elif chain_dir.is_dir():
        pdb_files = sorted(list(chain_dir.glob("*.pdb")) + list(chain_dir.glob("*.ent")))
    else:
        print(f"[PYTHON-INFO] Target directory/file not found: {chain_dir}")
        return
    print(f"[PYTHON-INFO] Scanning {len(pdb_files)} PDB chain(s) in {chain_dir} for mapping tables in {aln_dir}")
    totals = {
        "processed_files": 0,
        "skipped_files": 0,
        "total_residues": 0,
        "protein_residues": 0,
        "non_protein_removed": 0,
        "reindexed_residues": 0,
        "unmapped_removed": 0,
    }
    per_chain_reports = []
    for pdb_path in tqdm(pdb_files, desc="Reindexing chains"):
        table_path = Path(args.table) if args.table else find_mapping_file(pdb_path.stem, aln_dir)
        if not table_path or not table_path.is_file():
            totals["skipped_files"] += 1
            continue
        mapping = load_mapping_table(table_path)
        if not mapping:
            totals["skipped_files"] += 1
            continue
        pdb_text, stats = reindex_pdb(pdb_path, mapping)
        if not stats:
            totals["skipped_files"] += 1
            continue
        if args.output and Path(args.output).suffix:
            out_file = Path(args.output)
        else:
            out_file = clean_dir / f"{pdb_path.stem}_reindexed.pdb"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(pdb_text)
        totals["processed_files"] += 1
        for k in ["total_residues", "protein_residues", "non_protein_removed", "reindexed_residues", "unmapped_removed"]:
            totals[k] += stats[k]
        report_str = (
            f"{pdb_path.stem}: Total={stats['total_residues']} | "
            f"Protein={stats['protein_residues']} | "
            f"Reindexed={stats['reindexed_residues']} | "
            f"Unmapped Removed={stats['unmapped_removed']} | "
            f"Non-Protein Removed={stats['non_protein_removed']}"
        )
        per_chain_reports.append(report_str)
    print("\n[PYTHON-INFO] Detailed Per-Chain Reindexing Breakdown:")
    for report in per_chain_reports:
        print(f"  - {report}")
    summary_msg = (
        f"Reindexed {totals['processed_files']}/{len(pdb_files)} chains "
        f"({totals['skipped_files']} skipped). "
        f"Residues -> Total: {totals['total_residues']}, "
        f"Protein: {totals['protein_residues']}, "
        f"Reindexed: {totals['reindexed_residues']}, "
        f"Unmapped Removed: {totals['unmapped_removed']}, "
        f"Non-Protein Removed: {totals['non_protein_removed']}"
    )
    finish_log(start, summary_msg)

if __name__ == "__main__":
    main()