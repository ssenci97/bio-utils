#!/usr/bin/env python
"""Perform pairwise local sequence alignment between query sequences and reference sequences."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
from pathlib import Path
from Bio.Align import PairwiseAligner
import pandas as pd
from tqdm import tqdm

DEFAULT_ALN_OUTDIR = "data/aln"
DEFAULT_CHAIN_REFSEQS_OUTDIR = "data/chains_refseqs"
DEFAULT_CHAIN_SEQS_OUTDIR = "data/chains_seqs"
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

###################################### HELPERS AND ALIGNMENT
def parse_fasta_file(fpath):
    if not fpath.exists():
        return "", ""
    lines = fpath.read_text().splitlines()
    header, seq = "", []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            header = line[1:]
        elif line:
            seq.append(line)
    return "".join(seq), header

def resolve_input_files(inp_arg, default_dir):
    if not inp_arg:
        inp_arg = default_dir
    p = Path(inp_arg)
    if p.is_dir():
        return sorted(list(p.glob("*.fasta")) + list(p.glob("*.fa")))
    if p.is_file() and p.suffix.lower() == ".txt":
        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return [Path(f) for f in lines]
    paths = [Path(x.strip()) for x in str(inp_arg).split(",") if x.strip()]
    return [f for f in paths if f.is_file()]

def run_alignment(ref_seq, query_seq, ref_id, query_id):
    if not ref_seq or not query_seq:
        return None, []
    aligner = PairwiseAligner(mode="local", match_score=1, mismatch_score=-1, open_gap_score=-2, extend_gap_score=-0.5)
    alignments = aligner.align(ref_seq, query_seq)
    if not alignments:
        return None, []
    alignment = alignments[0]
    a, b = alignment
    ref_start, query_start = int(alignment.coordinates[0, 0]), int(alignment.coordinates[1, 0])
    ref_end, query_end = int(alignment.coordinates[0, 1]), int(alignment.coordinates[1, 1])
    matches = mismatches = insertions = deletions = 0
    gappos, q_pos, ref_pos = [], query_start, ref_start
    pos_rows = []
    for idx, (x, y) in enumerate(zip(a, b), start=1):
        if x != "-":
            ref_pos += 1
        if y != "-":
            q_pos += 1
        if x == "-" and y == "-":
            continue
        if x == "-":
            insertions += 1
        elif y == "-":
            deletions += 1
            gappos.append(str(q_pos))
        elif x == y:
            matches += 1
        else:
            mismatches += 1
        pos_rows.append({
            "alignment_position": idx, "ref_id": ref_id, "ref": x, "alt": y,
            "posref": ref_pos if x != "-" else "", "posalt": q_pos if y != "-" else "",
            "match": "1" if x == y and x != "-" else "0"
        })
    aligned_len = matches + mismatches + insertions + deletions
    q_aligned_len = len(query_seq) - deletions
    ident_pct = round(matches / aligned_len * 100, 2) if aligned_len else 0.0
    cov_pct = round(q_aligned_len / len(query_seq) * 100, 2)
    score = round(matches - 0.5 * mismatches - 1.5 * (insertions + deletions), 2)
    stats = {
        "query_id": query_id, "ref_id": ref_id, "query_length": len(query_seq), "ref_length": len(ref_seq),
        "aligned_length": aligned_len, "matches": matches, "mismatches": mismatches,
        "insertions_in_query": insertions, "deletions_in_query": deletions,
        "identity_percent": ident_pct, "query_coverage_percent": cov_pct, "alignment_score": score,
        "range": f"{query_start + 1}-{query_end}", "ref_start": ref_start, "ref_end": ref_end,
        "gappos": ",".join(gappos), "sequence": ref_seq[ref_start:ref_end]
    }
    return stats, pos_rows

def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Pairwise local sequence alignment")
    parser.add_argument("-i", "--input", help="Query sequence file path(s), txt list, or directory")
    parser.add_argument("-o", "--output", help="Alignment output directory (defaults to aln_outdir in config)")
    parser.add_argument("--sequence", help="Manual reference sequence string")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()
    defaults = {
        "chain_seqs_outdir": DEFAULT_CHAIN_SEQS_OUTDIR,
        "chain_refseqs_outdir": DEFAULT_CHAIN_REFSEQS_OUTDIR,
        "aln_outdir": DEFAULT_ALN_OUTDIR
    }
    cfg = resolve_config(args.json, defaults)
    query_files = resolve_input_files(args.input, cfg["chain_seqs_outdir"])
    ref_dir = Path(cfg["chain_refseqs_outdir"])
    aln_dir = Path(args.output) if args.output else Path(cfg["aln_outdir"])
    aln_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PYTHON-INFO] Aligning {len(query_files)} query sequence file(s) into {aln_dir}")
    count = 0
    for q_file in tqdm(query_files, desc="Aligning sequences"):
        q_seq, _ = parse_fasta_file(q_file) if q_file.suffix in [".fasta", ".fa"] else (q_file.read_text().strip(), "")
        if not q_seq:
            continue
        q_id = q_file.stem
        if args.sequence:
            r_seq, r_id = args.sequence.strip(), "manual_ref"
        else:
            r_file = ref_dir / f"{q_id}.fasta"
            r_seq, r_id = parse_fasta_file(r_file)
            if not r_seq:
                continue
        stats, pos_rows = run_alignment(r_seq, q_seq, r_id, q_id)
        if stats:
            pd.DataFrame([stats]).to_csv(aln_dir / f"{q_id}_stats.tsv", sep="\t", index=False)
            pd.DataFrame(pos_rows).to_csv(aln_dir / f"{q_id}_positions.tsv", sep="\t", index=False)
            count += 1
    finish_log(start, f"Completed {count} pairwise alignments out of {len(query_files)} queries")

if __name__ == "__main__":
    main()


