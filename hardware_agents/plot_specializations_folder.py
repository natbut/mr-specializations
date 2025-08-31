#!/usr/bin/env python3
"""
Plot per-agent specialization weights from a folder of CSV logs.

Update: Bars for rollout rows cycle through seaborn pastel colors consistently across agents.
Agents with even index (0,2,4,...) use solid fill; odd index (1,3,5,...) use hatch fill.
Thus, for two files: first agent = solid, second agent = hatched.

Expected CSV columns:
  - 'specializations' (or 'specializaions') column with list-like string of floats.

Outputs:
  - weights_<index>.png : For each rollout index, a figure with one subplot per agent.

"""
import argparse
import glob
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_list(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s.lower() in ('nan', 'none'):
        return None
    if '...' in s:
        return None
    try:
        val = literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
    except Exception:
        pass
    try:
        s2 = s.strip().lstrip('[').rstrip(']')
        if not s2:
            return []
        parts = [p.strip() for p in s2.split(',')]
        return [float(p) for p in parts if p]
    except Exception:
        return None

def load_agent_series(csv_path):
    df = pd.read_csv(csv_path)
    col = None
    if 'specializations' in df.columns:
        col = 'specializations'
    elif 'specializaions' in df.columns:
        col = 'specializaions'
    else:
        raise ValueError(f"{os.path.basename(csv_path)} missing 'specializations' column")
    weights_series = {}
    for idx, val in df[col].items():
        weights = parse_list(val)
        weights_series[idx] = weights
    timestamps = None
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].tolist()
    return weights_series, timestamps

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=str, required=True, help='Folder containing one or more CSV files')
    parser.add_argument('--rows', type=int, nargs=2, metavar=('START', 'END'),
                        help='Inclusive range of row indices (rollouts) to plot')
    parser.add_argument('--outdir', type=str, default='figs', help='Output directory')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    parser.add_argument('--pattern', type=str, default='*.csv', help='Glob pattern for CSV files in the folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not paths:
        raise SystemExit(f"No CSV files found in folder: {args.folder}")

    agents = []
    agent_weights = []
    agent_timestamps = []

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        series, ts = load_agent_series(p)
        agents.append(name)
        agent_weights.append(series)
        agent_timestamps.append(ts)

    all_indices_sets = [set(d.keys()) for d in agent_weights]
    common_indices = set.intersection(*all_indices_sets) if all_indices_sets else set()
    if args.rows:
        start, end = args.rows
        common_indices = {i for i in common_indices if start <= i <= end}
    indices = sorted(common_indices)
    if not indices:
        raise SystemExit("No common row indices to plot across agents.")

    # Determine maximum weight vector length
    max_len = 0
    for d in agent_weights:
        for i in indices:
            w = d.get(i)
            if isinstance(w, (list, tuple)):
                max_len = max(max_len, len(w))

    # Palette for row-wise coloring
    row_palette = sns.color_palette("pastel", len(indices))

    ROBOTS = ["BlueROV", "Lutra"]

    for i_idx, i in enumerate(indices):
        n = len(agents)
        fig_w = max(8, 3 * n)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 4), sharey=True)
        if n == 1:
            axes = [axes]
        row_color = row_palette[i_idx % len(row_palette)]

        for a_idx, (ax, agent, d) in enumerate(zip(axes, agents, agent_weights)):
            w = d.get(i)
            if not isinstance(w, (list, tuple)):
                w = []
            arr = np.array([float(v) for v in w] + [np.nan] * (max_len - len(w)))
            if a_idx % 2 == 0:
                ax.bar(np.arange(max_len), arr, width=0.8, color=row_color)
            else:
                ax.bar(np.arange(max_len), arr, width=0.8, color=row_color, hatch='//', edgecolor='white', linewidth=0)
            ax.set_title(ROBOTS[a_idx])
            ax.set_xlabel('Specialization Idx')
            ax.set_ylim([-1.0, 1.0])
            ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
            ax.set_xticks(np.arange(max_len))

        axes[0].set_ylabel('Scalar Value')

        suptitle = f"Rollout {i} — Specializations by Robot"
        ts_values = []
        for ts in agent_timestamps:
            if ts is not None and i < len(ts):
                ts_values.append(ts[i])
        if len(ts_values) == len(agents) and len(set(ts_values)) == 1:
            suptitle = f"{suptitle} (timestamp: {ts_values[0]})"
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        outpath = os.path.join(args.outdir, f"weights_{i}.png")
        fig.savefig(outpath, dpi=args.dpi)
        plt.close(fig)

if __name__ == '__main__':
    main()
