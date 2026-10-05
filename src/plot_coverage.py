#!/usr/bin/env python
"""Generate multi-page PDF coverage plots from local-alignment statistics."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import os
import re
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
ROW_HEIGHT_CM = 1.6
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
    df["pdb_id"] = parts[0]
    df["chain_id"] = parts[1] if parts.shape[1] > 1 else ""
    return df

def numeric_column(df, name, default=0):
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)

###################################### COVERAGE
def parse_gappos(gap_str):
    if pd.isna(gap_str) or not str(gap_str).strip():
        return set()
    gaps = set()
    for part in str(gap_str).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            p = part.split("-")
            try:
                gaps.update(range(int(p[0]), int(p[1]) + 1))
            except ValueError:
                pass
        else:
            try:
                gaps.add(int(part))
            except ValueError:
                pass
    return gaps

def coverage_ranges(row):
    seq = str(row.get("sequence", "")).strip() if pd.notna(row.get("sequence", "")) else ""
    intervals = []
    if seq and seq.upper() != "NAN":
        try:
            rs = int(row.get("ref_start", 1))
        except (ValueError, TypeError):
            rs = 1
        rs = max(1, rs)
        curr_pos = rs
        block_start = None
        for char in seq:
            if char != "-":
                if block_start is None:
                    block_start = curr_pos
                curr_pos += 1
            else:
                if block_start is not None:
                    intervals.append((block_start, curr_pos - 1))
                    block_start = None
        if block_start is not None:
            intervals.append((block_start, curr_pos - 1))
    if not intervals:
        range_val = row.get("range", "")
        if pd.notna(range_val) and str(range_val).strip():
            matches = re.findall(r"(\d+)\s*[\-\.\:]+\s*(\d+)", str(range_val))
            for s, e in matches:
                start, end = int(s), int(e)
                if start >= 0 and end >= start:
                    intervals.append((start, end))
    if not intervals:
        rs_val = row.get("ref_start", "")
        re_val = row.get("ref_end", "")
        rs_list = [int(x) for x in re.findall(r"\d+", str(rs_val))] if pd.notna(rs_val) else []
        re_list = [int(x) for x in re.findall(r"\d+", str(re_val))] if pd.notna(re_val) else []
        if rs_list and re_list and len(rs_list) == len(re_list):
            for s, e in zip(rs_list, re_list):
                if s >= 0 and e >= s:
                    intervals.append((s, e))
        elif len(rs_list) == 1 and len(re_list) == 1:
            s, e = rs_list[0], re_list[0]
            if s >= 0 and e >= s:
                intervals.append((s, e))
    if not intervals:
        return []
    gappos_val = row.get("gappos", "")
    gaps = parse_gappos(gappos_val)
    if not gaps:
        return intervals
    split_intervals = []
    for start, end in intervals:
        curr = start
        for pos in range(start, end + 1):
            if pos in gaps:
                if curr < pos:
                    split_intervals.append((curr, pos - 1))
                curr = pos + 1
        if curr <= end:
            split_intervals.append((curr, end))
    return split_intervals

def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def get_nice_step(xmax):
    target = 5
    raw = xmax / target
    mag = 10 ** np.floor(np.log10(max(raw, 1)))
    res = raw / mag
    if res < 1.5:
        factor = 1
    elif res < 3:
        factor = 2
    elif res < 7:
        factor = 5
    else:
        factor = 10
    return int(factor * mag)

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

    grouped_rows = []
    for (query_id, pdb_id, chain_id), group in df.groupby(["query_id", "pdb_id", "chain_id"], sort=False):
        ref_len = max(group["ref_length"].max(), 1)
        query_len = max(group["query_length"].max(), 1)
        identity = group["identity_percent"].mean()
        all_intervals = []
        for _, r in group.iterrows():
            all_intervals.extend(coverage_ranges(r))
        merged = merge_intervals(all_intervals)
        grouped_rows.append(
            {
                "query_id": query_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "ref_length": ref_len,
                "query_length": query_len,
                "identity_percent": identity,
                "method": "local alignment",
                "ranges": merged,
            }
        )
    df_plot = pd.DataFrame(grouped_rows)
    df_plot["sort_pdb"] = df_plot["pdb_id"].astype(str).str.upper()
    df_plot = df_plot.sort_values(
        by=["sort_pdb", "chain_id", "ref_length"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    total_records = len(df_plot)
    total_pages = (total_records + PAGE_SIZE - 1) // PAGE_SIZE
    methods = list(df_plot["method"].unique())
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
            chunk = df_plot.iloc[
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
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax_legend.legend(
                handles=handles,
                loc="center",
                ncol=min(3, len(handles)),
                frameon=False,
                fontsize=10,
            )
            for i, (_, row) in enumerate(chunk.iterrows()):
                ax = fig.add_subplot(gs[i + 2])
                pdb_id = str(row["pdb_id"]).upper()
                chain_id = str(row["chain_id"])
                ref_len = max(int(row["ref_length"]), 1)
                query_length = max(int(row["query_length"]), 1)
                identity = float(row["identity_percent"])
                ranges = row["ranges"]

                # Calculate total aligned coverage length
                coverage_length = sum(
                    (end - start) if start == 0 else (end - start + 1)
                    for start, end in ranges
                )
                ref_cov_pct = min(100.0, coverage_length / ref_len * 100) if ref_len else 0.0
                query_cov_pct = min(100.0, coverage_length / query_length * 100) if query_length else 0.0

                # Focus X-axis scale (window size <= 2 * total aligned span)
                if ranges:
                    align_min = min(s for s, e in ranges)
                    align_max = max(e for s, e in ranges)
                    aligned_len = max(align_max - align_min + 1, 1)
                    max_window = 2 * aligned_len
                    
                    if ref_len <= max_window:
                        xmin = 0.0
                        xmax = max(ref_len, 10) * 1.05
                    else:
                        center = (align_min + align_max) / 2.0
                        half_w = max_window / 2.0
                        xmin = max(0.0, center - half_w)
                        xmax = xmin + max_window
                        if xmax > ref_len:
                            xmax = float(ref_len)
                            xmin = max(0.0, xmax - max_window)
                else:
                    xmin = 0.0
                    xmax = max(ref_len, 50) * 1.05

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(-0.45, 1.45)
                ax.set_yticks([0.5])

                # All information organized inside Y tick label
                y_label = (
                    f"{pdb_id}:{chain_id}\n"
                    f"Ref: {ref_len} aa ({ref_cov_pct:.1f}% cov)\n"
                    f"Query: {query_length} aa ({query_cov_pct:.1f}% cov, {identity:.1f}% id)"
                )
                ax.set_yticklabels([y_label], fontsize=8.5)

                # Reference sequence hatched background bar
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
                
                # Aligned region blocks
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

                step = get_nice_step(xmax - xmin)
                start_tick = np.ceil(xmin / step) * step if xmin > 0 else 0
                ticks = np.arange(start_tick, xmax + step * 0.01, step)
                ticks = [t for t in ticks if xmin <= t <= xmax]
                tick_labels = [str(int(t)) if t != 0 else "" for t in ticks]
                
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
                mid_step = step / 2
                mid_ticks = np.arange(xmin + mid_step, xmax, step)
                mid_ticks = [t for t in mid_ticks if xmin <= t <= xmax]
                ax.set_xticks(mid_ticks, minor=True)

                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                )
                ax.tick_params(
                    axis="x",
                    which="major",
                    labelsize=8.5,
                    pad=2,
                )
                for x in ticks:
                    if x > xmin and x < xmax:
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
                        fontsize=11,
                        fontweight="bold",
                    )
            fig.subplots_adjust(
                left=0.32,
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