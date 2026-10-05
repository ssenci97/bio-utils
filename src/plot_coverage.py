#!/usr/bin/env python
"""Generate multi-page PDF coverage plots from local-alignment statistics."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import os
import resource
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

A4_WIDTH_IN = 8.27
DEFAULT_ALN_OUTDIR = "data/chains/aln"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_OUTPUT = "coverage.pdf"
DEFAULT_PLOTS_OUTDIR = "data/plots"
LEGEND_HEIGHT_CM = 1.2
PAGE_SIZE = 100
ROW_HEIGHT_CM = 1.4
SEED = 42
TITLE_HEIGHT_CM = 0.6

###################################### CONFIG
def resolve_config(json_arg, defaults, no_json):
    if no_json:
        return dict(defaults)
    path = Path(json_arg)
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}
    changed = False
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
            changed = True
    if changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    return {key: cfg[key] for key in defaults}

###################################### INPUT
def load_stats(aln_dir):
    path = Path(aln_dir)
    files = [path] if path.is_file() else sorted(path.glob("*_stats.tsv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as exc:
            print(f"[PYTHON-INFO] Could not read {f}: {exc}")
            continue
        if df.empty:
            continue
        df["stats_file"] = f.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "query_id" not in df.columns:
        df["query_id"] = df["stats_file"].str.replace("_stats.tsv", "", regex=False)
    parts = df["query_id"].astype(str).str.rsplit("_", n=1, expand=True)
    df["pdb_id"] = parts[0].str.lower()
    df["chain_id"] = parts[1] if parts.shape[1] > 1 else ""
    return df

def numeric_column(df, name, default=0):
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)

###################################### COVERAGE
def coverage_ranges(row):
    try:
        start = int(row["ref_start"])
        end = int(row["ref_end"])
    except (KeyError, TypeError, ValueError):
        return []
    if start < 0 or end < start:
        return []
    return [(start, end)]

def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

###################################### PLOTTING
def generate_plot(df, output_pdf):
    if df.empty:
        print("[PYTHON-INFO] No alignment statistics found; plot skipped")
        return 0
    df = df.copy()
    df["ref_length"] = numeric_column(df, "ref_length", 0).astype(int)
    df["ref_start"] = numeric_column(df, "ref_start", 0).astype(int)
    df["ref_end"] = numeric_column(df, "ref_end", 0).astype(int)
    df["query_length"] = numeric_column(df, "query_length", 0).astype(int)
    df["identity_percent"] = numeric_column(df, "identity_percent", 0)
    df["method"] = "local alignment"
    df = df.sort_values(
        by=["pdb_id", "chain_id", "ref_length"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    total_records = len(df)
    total_pages = (total_records + PAGE_SIZE - 1) // PAGE_SIZE
    methods = list(df["method"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {method: cmap(i % 10) for i, method in enumerate(methods)}
    handles = [
        patches.Patch(
            facecolor=color_map[method],
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )
        for method in methods
    ]
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for page_idx in tqdm(range(total_pages), desc="Plotting PDF pages"):
            chunk = df.iloc[
                page_idx * PAGE_SIZE : (page_idx + 1) * PAGE_SIZE
            ].reset_index(drop=True)
            n = len(chunk)
            fig_h = (
                TITLE_HEIGHT_CM + LEGEND_HEIGHT_CM + n * ROW_HEIGHT_CM
            ) / 2.54
            fig = plt.figure(figsize=(A4_WIDTH_IN, fig_h))
            gs = fig.add_gridspec(
                n + 2,
                1,
                height_ratios=[TITLE_HEIGHT_CM, LEGEND_HEIGHT_CM]
                + [ROW_HEIGHT_CM] * n,
                hspace=0.85,
            )
            ax_title = fig.add_subplot(gs[0])
            ax_legend = fig.add_subplot(gs[1])
            ax_title.axis("off")
            ax_legend.axis("off")
            title = (
                f"PDB Chain Coverage (Page {page_idx + 1}/{total_pages})"
                if total_pages > 1
                else "PDB Chain Coverage"
            )
            ax_title.text(
                0.5,
                0.5,
                title,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax_legend.legend(
                handles=handles,
                loc="center",
                ncol=min(3, len(handles)),
                frameon=False,
                fontsize=7,
            )
            for i, (_, row) in enumerate(chunk.iterrows()):
                ax = fig.add_subplot(gs[i + 2])
                pdb_id = str(row["pdb_id"])
                chain_id = str(row["chain_id"])
                ref_len = max(int(row["ref_length"]), 1)
                ranges = merge_intervals(coverage_ranges(row))
                xmax = max(ref_len, 50) * 1.05
                ax.set_xlim(0, xmax)
                ax.set_ylim(-0.45, 1.45)
                ax.set_yticks([0.5])
                identity = float(row["identity_percent"])
                query_length = int(row["query_length"])
                label = f"{pdb_id}:{chain_id} ({query_length} aa, {identity:.1f}%)"
                ax.set_yticklabels([label], fontsize=7)
                ax.add_patch(
                    patches.Rectangle(
                        (1, 0.2),
                        max(0, ref_len - 1),
                        0.6,
                        facecolor="none",
                        edgecolor="gray",
                        hatch="\\\\",
                        linewidth=0.6,
                    )
                )
                color = color_map["local alignment"]
                for start, end in ranges:
                    p_start = max(1, start)
                    p_end = max(p_start, end)
                    width = p_end - p_start if start == 0 else max(0, p_end - p_start)
                    ax.add_patch(
                        patches.Rectangle(
                            (p_start, 0.2),
                            width,
                            0.6,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.9,
                        )
                    )
                coverage_length = sum(
                    (end - start) if start == 0 else (end - start + 1)
                    for start, end in ranges
                )
                coverage_pct = (
                    min(100.0, coverage_length / ref_len * 100)
                    if ref_len
                    else 0.0
                )
                ax.text(
                    0.995,
                    0.88,
                    f"{coverage_pct:.1f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6,
                )
                step = 50 if xmax < 300 else 100 if xmax < 800 else 200
                ticks = np.arange(0, xmax, step)
                ax.set_xticks(ticks)
                ax.tick_params(
                    axis="x",
                    bottom=True,
                    labelbottom=True,
                    labelsize=6,
                    pad=2,
                )
                for x in ticks[1:]:
                    ax.axvline(
                        x=x,
                        linestyle="--",
                        alpha=0.35,
                        linewidth=0.7,
                    )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if i == n - 1:
                    ax.set_xlabel(
                        "Reference Sequence Position",
                        fontsize=8,
                        fontweight="bold",
                    )
            fig.subplots_adjust(
                left=0.18,
                right=0.96,
                top=0.98,
                bottom=0.06,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return total_records

###################################### LOGGING
def finish_log(start, message):
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[PYTHON-INFO] {now} | {message} | "
        f"execution time: {elapsed:.2f}s | max memory: {memory_mb:.1f} MB"
    )

###################################### MAIN EXECUTION
def main():
    np.random.seed(SEED)
    start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Plot PDB chain coverage from alignment statistics"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Alignment statistics directory or *_stats.tsv file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Coverage PDF path",
    )
    parser.add_argument(
        "-j",
        "--json",
        nargs="?",
        const=DEFAULT_CONFIG_PATH,
        default=DEFAULT_CONFIG_PATH,
        help="JSON configuration path",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Use built-in defaults without reading or updating JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing coverage PDF",
    )
    args = parser.parse_args()
    defaults = {
        "aln_outdir": DEFAULT_ALN_OUTDIR,
        "plots_outdir": DEFAULT_PLOTS_OUTDIR,
        "coverage_pdf": DEFAULT_OUTPUT,
    }
    cfg = resolve_config(args.json, defaults, args.no_json)
    input_path = Path(args.input or cfg["aln_outdir"])
    output_pdf = Path(
        args.output or Path(cfg["plots_outdir"]) / cfg["coverage_pdf"]
    )
    if output_pdf.exists() and not args.force:
        print(f"[PYTHON-INFO] Coverage PDF already exists: {output_pdf}; skipped")
        finish_log(start, "Coverage plot skipped")
        return
    df = load_stats(input_path)
    if df.empty:
        raise FileNotFoundError(
            f"[PYTHON-INFO] No *_stats.tsv files found in {input_path}"
        )
    records = generate_plot(df, output_pdf)
    finish_log(
        start,
        f"Saved {output_pdf} from {records} alignment statistics",
    )

if __name__ == "__main__":
    main()