#!/usr/bin/env python
"""Compute residue-residue contact maps in nanometers for cleaned PDB files grouped by PDB ID."""
###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CONTACT_TYPE = "inter"
DEFAULT_CONTACTS_OUTDIR = "data/contacts"
DEFAULT_CUTOFF = 0.8
SEED = 42
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
###################################### CONTACT LOGIC
def extract_pdb_id(filename_stem):
    return filename_stem.split("_")[0].lower()
def load_chain_traj(pdb_path):
    try:
        traj = md.load(str(pdb_path))
    except Exception as e:
        print(f"[PYTHON-INFO] Failed to parse {pdb_path}: {e}")
        return None, None
    stem_parts = pdb_path.stem.split("_")
    chain_id = stem_parts[1] if len(stem_parts) > 1 else "A"
    if traj.topology.n_chains == 0:
        return chain_id, traj
    for chain in traj.topology.chains:
        cid = chain.chain_id if chain.chain_id.strip() else chain_id
        return cid, traj
    return chain_id, traj
def calculate_contacts_for_pdb(chain_data_list, cutoff=0.8, contact_type="inter"):
    contacts = []
    calc_intra = contact_type in ("intra", "all")
    calc_inter = contact_type in ("inter", "all")
    for i in range(len(chain_data_list)):
        cid1, traj1 = chain_data_list[i]
        top1 = traj1.topology
        if calc_intra and top1.n_residues > 1:
            pairs = []
            for r1 in range(top1.n_residues):
                for r2 in range(r1 + 1, top1.n_residues):
                    res1 = top1.residue(r1)
                    res2 = top1.residue(r2)
                    if abs(res1.resSeq - res2.resSeq) < 1:
                        continue
                    pairs.append([r1, r2])
            if pairs:
                dists, _ = md.compute_contacts(traj1, contacts=np.array(pairs), scheme="ca", ignore_nonprotein=False, periodic=False)
                for k, (r1_idx, r2_idx) in enumerate(pairs):
                    min_d = float(dists[0, k])
                    if min_d <= cutoff:
                        res1 = top1.residue(r1_idx)
                        res2 = top1.residue(r2_idx)
                        contacts.append({
                            "chain_1": cid1,
                            "resnum_1": res1.resSeq,
                            "resname_1": res1.name.strip(),
                            "chain_2": cid1,
                            "resnum_2": res2.resSeq,
                            "resname_2": res2.name.strip(),
                            "min_distance_nm": round(min_d, 4),
                            "contact_type": "intra",
                        })
        if calc_inter:
            for j in range(i + 1, len(chain_data_list)):
                cid2, traj2 = chain_data_list[j]
                top2 = traj2.topology
                combined = traj1.stack(traj2)
                n1 = top1.n_residues
                pairs = [[r1, n1 + r2] for r1 in range(n1) for r2 in range(top2.n_residues)]
                if pairs:
                    dists, _ = md.compute_contacts(combined, contacts=np.array(pairs), scheme="ca", ignore_nonprotein=False, periodic=False)
                    for k, (r1_idx, r2_idx) in enumerate(pairs):
                        min_d = float(dists[0, k])
                        if min_d <= cutoff:
                            res1 = top1.residue(r1_idx)
                            res2 = top2.residue(r2_idx - n1)
                            contacts.append({
                                "chain_1": cid1,
                                "resnum_1": res1.resSeq,
                                "resname_1": res1.name.strip(),
                                "chain_2": cid2,
                                "resnum_2": res2.resSeq,
                                "resname_2": res2.name.strip(),
                                "min_distance_nm": round(min_d, 4),
                                "contact_type": "inter",
                            })
    return contacts
def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")
###################################### MAIN EXECUTION
def main():
    np.random.seed(SEED)
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Compute contact maps across cleaned PDB chains grouped by PDB ID")
    parser.add_argument("-i", "--input", help="Clean PDB file path or directory (defaults to clean_outdir in config)")
    parser.add_argument("-o", "--output", help="Output directory for contact TSVs (defaults to contacts_outdir in config)")
    parser.add_argument("-d", "--cutoff", type=float, help="Heavy atom distance cutoff threshold in nanometers (default: 0.8 nm)")
    parser.add_argument("--type", choices=["inter", "intra", "all"], default=DEFAULT_CONTACT_TYPE, help="Type of contacts to calculate: 'inter' (default), 'intra', or 'all'")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    args = parser.parse_args()
    defaults = {"clean_outdir": DEFAULT_CLEAN_OUTDIR, "contact_type": DEFAULT_CONTACT_TYPE, "contacts_outdir": DEFAULT_CONTACTS_OUTDIR, "distance_threshold": DEFAULT_CUTOFF}
    cfg = resolve_config(args.json, defaults)
    clean_dir = Path(args.input) if args.input else Path(cfg["clean_outdir"])
    contacts_dir = Path(args.output) if args.output else Path(cfg["contacts_outdir"])
    cutoff = args.cutoff if args.cutoff is not None else float(cfg.get("distance_threshold", DEFAULT_CUTOFF))
    contact_type = args.type if args.type != DEFAULT_CONTACT_TYPE else cfg.get("contact_type", DEFAULT_CONTACT_TYPE)
    contacts_dir.mkdir(parents=True, exist_ok=True)
    if clean_dir.is_file():
        pdb_files = [clean_dir]
    elif clean_dir.is_dir():
        pdb_files = sorted(list(clean_dir.glob("*.pdb")) + list(clean_dir.glob("*.ent")))
    else:
        print(f"[PYTHON-INFO] Target directory/file not found: {clean_dir}")
        return
    groups = defaultdict(list)
    for fpath in pdb_files:
        pdb_id = extract_pdb_id(fpath.stem)
        groups[pdb_id].append(fpath)
    print(f"[PYTHON-INFO] Found {len(pdb_files)} PDB chain(s) spanning {len(groups)} unique PDB ID(s) in {clean_dir}")
    print(f"[PYTHON-INFO] Computing '{contact_type}' contacts (cutoff: {cutoff} nm) into {contacts_dir}")
    total_contacts = 0
    total_inter = 0
    total_intra = 0
    for pdb_id, files in tqdm(groups.items(), desc="Calculating contacts"):
        chain_data_list = []
        for fpath in files:
            cid, traj = load_chain_traj(fpath)
            if traj is not None and traj.topology.n_residues > 0:
                chain_data_list.append((cid, traj))
        if not chain_data_list:
            continue
        contacts = calculate_contacts_for_pdb(chain_data_list, cutoff=cutoff, contact_type=contact_type)
        if contacts:
            df = pd.DataFrame(contacts)
            df.insert(0, "pdb_id", pdb_id)
            out_file = contacts_dir / f"{pdb_id}_contacts_{contact_type}.tsv"
            df.to_csv(out_file, sep="\t", index=False)
            total_contacts += len(df)
            total_inter += len(df[df["contact_type"] == "inter"])
            total_intra += len(df[df["contact_type"] == "intra"])
    finish_log(start, f"Saved contacts for {len(groups)} PDB IDs -> Total: {total_contacts} (Inter-chain: {total_inter}, Intra-chain: {total_intra}) at ≤{cutoff} nm")
if __name__ == "__main__":
    main()