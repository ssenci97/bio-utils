#!/usr/bin/env python
"""Generate sequence contact string CSV for cleaned PDB chains using alignment position files."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
from pathlib import Path

from Bio import SeqIO
import pandas as pd
from tqdm import tqdm

DEFAULT_ALN_DIR = "data/aln"
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CONTACT_TYPE = "inter"
DEFAULT_CONTACTS_OUTDIR = "data/contacts"
DEFAULT_OUTPUT_PATH = "data/contacts/contacts_sequences.csv"
DEFAULT_REFSEQS_DIR = "data/refseqs"
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

###################################### REFSEQ LOADER
def load_all_refseqs(*search_paths):
    refseq_map = {}
    for path_str in search_paths:
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            continue
        files = [p] if p.is_file() else list(p.rglob("*.fasta")) + list(p.rglob("*.fa")) + list(p.rglob("*.faa"))
        for f in files:
            try:
                for record in SeqIO.parse(f, "fasta"):
                    seq_str = str(record.seq).upper()
                    rec_id = record.id.strip().lower()
                    refseq_map[rec_id] = seq_str
                    stem_clean = f.stem.replace("_refseq", "").replace("_positions", "").lower()
                    refseq_map[stem_clean] = seq_str
            except Exception:
                continue
    return refseq_map

###################################### ALIGNMENT AND CONTACT PARSER
def parse_alignment_file(aln_path):
    try:
        df = pd.read_csv(aln_path, sep=r"\s+|\t|,", engine="python")
    except Exception:
        return None
    if not {"posref", "posalt"}.issubset(set(df.columns)):
        return None
    df["posref"] = pd.to_numeric(df["posref"], errors="coerce")
    df["posalt"] = pd.to_numeric(df["posalt"], errors="coerce")
    ref_df = df.dropna(subset=["posref"]).copy()
    if ref_df.empty:
        return None
    ref_df["posref"] = ref_df["posref"].astype(int)
    stem = aln_path.stem.replace("_positions", "")
    parts = stem.split("_")
    pdb_id = parts[0].upper()
    chain_id = parts[1] if len(parts) > 1 else "A"
    min_posref = int(ref_df["posref"].min())
    max_posref = int(ref_df["posref"].max())
    ref_seq_dict = {}
    if "ref" in ref_df.columns:
        for _, row in ref_df.iterrows():
            ref_seq_dict[int(row["posref"])] = str(row["ref"]).upper()
    pdb_df = ref_df.dropna(subset=["posalt"]).copy()
    pdb_df["posalt"] = pdb_df["posalt"].astype(int)
    pdb_ref_pos = set(pdb_df["posref"])
    return {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "chain_key": f"{pdb_id}_{chain_id}".lower(),
        "min_posref": min_posref,
        "max_posref": max_posref,
        "aln_df": pdb_df,
        "pdb_ref_pos": pdb_ref_pos,
        "ref_seq_dict": ref_seq_dict,
    }

def load_contacts_for_pdb(contacts_dir, pdb_id, contact_type="inter"):
    pdb_lower = pdb_id.lower()
    candidates = [
        contacts_dir / f"{pdb_lower}_contacts_{contact_type}.tsv",
        contacts_dir / f"{pdb_lower}_contacts_{contact_type}.csv",
        contacts_dir / f"{pdb_id}_contacts_{contact_type}.tsv",
        contacts_dir / f"{pdb_id}_contacts_{contact_type}.csv",
        contacts_dir / f"{pdb_lower}_contacts.tsv",
        contacts_dir / f"{pdb_lower}_contacts.csv",
        contacts_dir / f"{pdb_id}_contacts.tsv",
        contacts_dir / f"{pdb_id}_contacts.csv",
        contacts_dir / f"{pdb_lower}.tsv",
        contacts_dir / f"{pdb_lower}.csv",
        contacts_dir / f"{pdb_id}.tsv",
        contacts_dir / f"{pdb_id}.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                return pd.read_csv(cand, sep=r"\s+|\t|,", engine="python")
            except Exception:
                pass
    return None

def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")

###################################### DEDUPLICATION AND MERGING FORMATTING
def format_complex_and_id(pdb_chains):
    """
    Format unique upper-case complex IDs and tidy ID string.
    pdb_chains: dict mapping pdb_id -> list/set of chain_ids
    Returns: (complex_str, id_str)
    e.g. complex_str = "1ABC_2XYZ"
         id_str      = "1ABC-AB_2XYZ-C"
    """
    sorted_pdbs = sorted(set(p.upper() for p in pdb_chains.keys()))
    complex_str = "_".join(sorted_pdbs)
    
    id_parts = []
    for pdb_id in sorted_pdbs:
        chains = []
        for k, v in pdb_chains.items():
            if k.upper() == pdb_id:
                chains.extend(v)
        unique_chains = sorted(set(chains))
        chains_str = "".join(unique_chains)
        id_parts.append(f"{pdb_id}-{chains_str}")
    id_str = "_".join(id_parts)
    
    return complex_str, id_str

def merge_pdb_chains(dict1, dict2):
    """Merge two pdb_chains dictionaries preserving unique chain IDs per uppercase PDB."""
    merged = {}
    for d in (dict1, dict2):
        for pdb_id, chains in d.items():
            p_upper = pdb_id.upper()
            if p_upper not in merged:
                merged[p_upper] = []
            for c in chains:
                if c not in merged[p_upper]:
                    merged[p_upper].append(c)
    return merged

def align_and_merge(r1, r2):
    s1, c1 = r1["sequence"], r1["contacts"]
    s2, c2 = r2["sequence"], r2["contacts"]
    l1, l2 = len(s1), len(s2)
    best_off = None
    for off in range(-20, 21):
        st1, en1 = max(0, off), min(l1, l2 + off)
        st2, en2 = max(0, -off), min(l2, l1 - off)
        if en1 <= st1:
            continue
        match = True
        for i, j in zip(range(st1, en1), range(st2, en2)):
            if s1[i] != "X" and s2[j] != "X" and s1[i] != s2[j]:
                match = False
                break
        if match:
            best_off = off
            break
    if best_off is None:
        return None
    off = best_off
    shift1 = -min(0, off)
    shift2 = off - min(0, off)
    new_len = max(l1 + shift1, l2 + shift2)
    seq_chars, con_chars = [], []
    for k in range(new_len):
        i1, i2 = k - shift1, k - shift2
        ch1 = s1[i1] if 0 <= i1 < l1 else "X"
        ch2 = s2[i2] if 0 <= i2 < l2 else "X"
        seq_chars.append(ch1 if ch1 != "X" else ch2)
        ct1 = c1[i1] if 0 <= i1 < l1 else "X"
        ct2 = c2[i2] if 0 <= i2 < l2 else "X"
        if ct1 == "1" or ct2 == "1":
            con_chars.append("1")
        elif ct1 == "0" or ct2 == "0":
            con_chars.append("0")
        else:
            con_chars.append("X")
    return "".join(seq_chars), "".join(con_chars)

def deduplicate_records(records, dedup_max=False):
    groups = {}
    if dedup_max:
        groups["ALL"] = records
    else:
        for r in records:
            groups.setdefault(r["complex"], []).append(r)
    final_rows = []
    for grp_recs in tqdm(groups.values(), desc="Deduplicating clusters"):
        active = grp_recs[:]
        changed = True
        while changed:
            changed = False
            new_active = []
            skip = set()
            for i in range(len(active)):
                if i in skip:
                    continue
                curr = active[i]
                for j in range(i + 1, len(active)):
                    if j in skip:
                        continue
                    target = active[j]
                    merged_res = align_and_merge(curr, target)
                    if merged_res is not None:
                        m_seq, m_con = merged_res
                        merged_chains = merge_pdb_chains(curr["pdb_chains"], target["pdb_chains"])
                        new_cmplx, new_id = format_complex_and_id(merged_chains)
                        curr = {
                            "complex": new_cmplx,
                            "id": new_id,
                            "sequence": m_seq,
                            "contacts": m_con,
                            "pdb_chains": merged_chains,
                        }
                        skip.add(j)
                        changed = True
                new_active.append(curr)
            active = new_active
        final_rows.extend(active)
    return final_rows

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build contacts-sequence 3-state CSV using alignment positions data")
    parser.add_argument("-i", "--input", "-a", "--aln", help="Alignment directory containing _positions files (defaults to aln_outdir in config)")
    parser.add_argument("-c", "--contacts", help="Contacts directory (defaults to contacts_outdir in config)")
    parser.add_argument("-r", "--refseqs", help="Optional reference sequences directory")
    parser.add_argument("-o", "--output", help="Output CSV path (default: data/contacts/contacts_sequences_{type}.csv)")
    parser.add_argument(
        "--type",
        choices=["inter", "intra", "all"],
        default=DEFAULT_CONTACT_TYPE,
        help="Type of contacts to process: 'inter' (default), 'intra', or 'all'",
    )
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--dedup", action="store_true", help="Group by sequence within same PDB and concatenate chain IDs")
    parser.add_argument("--dedup-max", action="store_true", help="Group by sequence bypassing chain IDs across all PDBs")
    args = parser.parse_args()

    defaults = {
        "aln_outdir": DEFAULT_ALN_DIR,
        "clean_outdir": DEFAULT_CLEAN_OUTDIR,
        "contact_type": DEFAULT_CONTACT_TYPE,
        "contacts_outdir": DEFAULT_CONTACTS_OUTDIR,
        "contacts_seq_output": DEFAULT_OUTPUT_PATH,
        "refseqs_dir": DEFAULT_REFSEQS_DIR,
    }
    cfg = resolve_config(args.json, defaults)

    aln_dir = Path(args.input) if args.input else Path(cfg.get("aln_outdir", DEFAULT_ALN_DIR))
    contacts_dir = Path(args.contacts) if args.contacts else Path(cfg["contacts_outdir"])
    refseq_dir = Path(args.refseqs) if args.refseqs else Path(cfg.get("refseqs_dir", DEFAULT_REFSEQS_DIR))
    contact_type = args.type if args.type != DEFAULT_CONTACT_TYPE else cfg.get("contact_type", DEFAULT_CONTACT_TYPE)

    if args.output:
        out_file = Path(args.output)
    elif "contacts_seq_output" in cfg and cfg["contacts_seq_output"] != DEFAULT_OUTPUT_PATH:
        out_file = Path(cfg["contacts_seq_output"])
    else:
        out_file = contacts_dir / f"contacts_sequences_{contact_type}.csv"

    tag = "_dedup_max" if args.dedup_max else ("_dedup" if args.dedup else "")
    if tag and not out_file.stem.endswith(tag):
        out_file = out_file.with_name(f"{out_file.stem}{tag}{out_file.suffix}")

    if out_file.suffix.lower() != ".csv":
        out_file = out_file.with_suffix(".csv")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    refseq_map = load_all_refseqs(refseq_dir, aln_dir)
    print(f"[PYTHON-INFO] Loaded {len(refseq_map)} sequence records from {refseq_dir} and {aln_dir}")

    if aln_dir.is_file():
        aln_files = [aln_dir]
    elif aln_dir.is_dir():
        pos_files = list(aln_dir.glob("*_positions.tsv")) + list(aln_dir.glob("*_positions.csv"))
        if not pos_files:
            pos_files = [f for f in aln_dir.glob("*.tsv") if not f.name.endswith("_stats.tsv")] + \
                        [f for f in aln_dir.glob("*.csv") if not f.name.endswith("_stats.csv")]
        aln_files = sorted(set(pos_files))
    else:
        print(f"[PYTHON-INFO] Alignment path not found: {aln_dir}")
        return

    print(f"[PYTHON-INFO] Found {len(aln_files)} alignment file(s) in {aln_dir}")

    rows = []
    processed_keys = set()

    for aln_path in tqdm(aln_files, desc="Building contact sequences"):
        info = parse_alignment_file(aln_path)
        if not info:
            continue

        pdb_id = info["pdb_id"]
        chain_id = info["chain_id"]
        chain_key = info["chain_key"]

        if chain_key in processed_keys:
            continue

        min_pos = info["min_posref"]
        max_pos = info["max_posref"]
        aln_df = info["aln_df"]
        pdb_ref_pos = info["pdb_ref_pos"]
        ref_seq_dict = info["ref_seq_dict"]

        ref_contact_pos = set()
        contacts_df = load_contacts_for_pdb(contacts_dir, pdb_id, contact_type)

        if contacts_df is not None and not contacts_df.empty:
            contacts_df.columns = [str(c).lower().strip() for c in contacts_df.columns]

            c1_col = next((c for c in ["chain_1", "chain1", "chain_a"] if c in contacts_df.columns), None)
            c2_col = next((c for c in ["chain_2", "chain2", "chain_b"] if c in contacts_df.columns), None)
            r1_col = next((c for c in ["resnum_1", "resnum1", "res_1", "resnum_a"] if c in contacts_df.columns), None)
            r2_col = next((c for c in ["resnum_2", "resnum2", "res_2", "resnum_b"] if c in contacts_df.columns), None)

            structure_contact_pos = set()
            target_chain = str(chain_id).strip()

            if c1_col and r1_col:
                mask1 = contacts_df[c1_col].astype(str).str.strip() == target_chain
                res1 = pd.to_numeric(contacts_df.loc[mask1, r1_col], errors="coerce").dropna().astype(int)
                structure_contact_pos.update(res1)

            if c2_col and r2_col:
                mask2 = contacts_df[c2_col].astype(str).str.strip() == target_chain
                res2 = pd.to_numeric(contacts_df.loc[mask2, r2_col], errors="coerce").dropna().astype(int)
                structure_contact_pos.update(res2)

            if structure_contact_pos:
                ref_contact_pos = set(
                    aln_df[aln_df["posalt"].isin(structure_contact_pos)]["posref"].astype(int)
                )

        full_refseq = refseq_map.get(chain_key) or refseq_map.get(pdb_id.lower()) or refseq_map.get(pdb_id)

        if full_refseq:
            start_idx = max(0, min_pos - 1)
            end_idx = min(len(full_refseq), max_pos)
            extracted_sequence = full_refseq[start_idx:end_idx]
        else:
            extracted_sequence = "".join(
                ref_seq_dict.get(pos, "X") for pos in range(min_pos, max_pos + 1)
            )

        target_len = max_pos - min_pos + 1
        if len(extracted_sequence) < target_len:
            extracted_sequence = extracted_sequence.ljust(target_len, "X")
        elif len(extracted_sequence) > target_len:
            extracted_sequence = extracted_sequence[:target_len]

        contact_bits = []
        for pos in range(min_pos, max_pos + 1):
            if pos in ref_contact_pos:
                contact_bits.append("1")
            elif pos in pdb_ref_pos:
                contact_bits.append("0")
            else:
                contact_bits.append("X")

        contacts_str = "".join(contact_bits)

        pdb_chains = {pdb_id: [chain_id]}
        cmplx, rec_id = format_complex_and_id(pdb_chains)

        rows.append({
            "complex": cmplx,
            "id": rec_id,
            "sequence": extracted_sequence,
            "contacts": contacts_str,
            "pdb_chains": pdb_chains,
        })
        processed_keys.add(chain_key)

    if args.dedup_max:
        rows = deduplicate_records(rows, dedup_max=True)
    elif args.dedup:
        rows = deduplicate_records(rows, dedup_max=False)

    if rows:
        out_df = pd.DataFrame(rows)[["complex", "id", "sequence", "contacts"]]
        out_df.to_csv(out_file, index=False)
        summary_msg = f"Saved {len(out_df)} chain contact sequences to CSV format at {out_file}"
    else:
        summary_msg = f"No chains found to write to {out_file}"

    finish_log(start, summary_msg)

if __name__ == "__main__":
    main()