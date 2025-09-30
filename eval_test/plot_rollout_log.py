#!/usr/bin/env python3
"""
Generate map and weights figures from a rollout CSV log.

CSV rows:
    rollout,step,entity_type,entity_id,x,y

Where "entity_type" includes:
  - 'base', 'obstacle', 'task' for static entities
  - 'agent_{i}_traj_{j}' for agent trajectories (connected polyline over steps)
  - 'weights' for per-agent weight vectors at a given rollout (entity_id == agent index).
    The x column contains a Python list string (e.g., "[0.1, -0.2, ...]").

Args:
  --csv PATH                  : path to the CSV log
  --rollouts START END        : (optional) inclusive range of rollout indices to plot. Example: --rollouts 0 10
  --outdir PATH               : (optional) output directory (default: ./figs)
  --dpi INT                   : (optional) figure DPI (default: 150)
  --show                      : (optional) show figures in a window (in addition to saving)

Outputs:
  For each rollout R in the selected range, produces:
   - map_R.png        : map figure with base/obstacle/task markers and agent trajectories
   - weights_R.png    : multi-axis figure with one subplot per agent showing that agent's weights at rollout R

Notes:
  - The parser skips repeated header lines.
  - The parser is robust to commas in the "weights" list by splitting each line at most 4 times.
  - If a weights row contains truncated strings (e.g., with "..."), the row is skipped.
  - Base and obstacles are aggregated globally across the entire CSV and drawn on every map figure.
  - Agents are color-coded (seaborn "pastel" palette) consistently across the map and weights for a given rollout.
"""
import argparse
import os
import re
from ast import literal_eval
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---- Helpers ----------------------------------------------------------------

# Use a few distinct markers for entities; extend as needed.
ENTITY_MARKERS = {
    'base': 'P',       # plus-filled star
    'obstacle': 's',   # X
    'task': 'X',       # triangle up
}

def parse_agent_traj(entity_type: str):
    """Parse strings like 'agent_2_traj_0' -> (agent_id=2, traj_id=0)."""
    m = re.match(r'^agent_(\d+)_traj_(\d+)$', entity_type)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def safe_parse_weights(weights_str: str):
    """Attempt to parse a Python-list-like string of floats; returns list or None. Skips strings with '...'."""
    if weights_str is None:
        return None
    s = weights_str.strip()
    if '...' in s:
        return None  # can't reconstruct truncated values robustly
    try:
        val = literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
        return None
    except Exception:
        return None

# ---- Data containers ---------------------------------------------------------

class RolloutData:
    def __init__(self):
        # Entities: dict[etype] -> list of (x,y)
        self.entities = defaultdict(list)
        # Trajectories: dict[(agent_id, traj_id)] -> list of (step, x, y)
        self.trajs = defaultdict(list)
        # Weights: dict[agent_id] -> list of floats
        self.weights = {}

def load_log(csv_path: str, r_start: int | None, r_end: int | None):
    """
    Load and group the log by rollout.
    Returns:
      rollouts: OrderedDict[int, RolloutData]
      global_entities: dict with keys 'base' and 'obstacle' aggregated across the entire CSV
    """
    rollouts: "OrderedDict[int, RolloutData]" = OrderedDict()
    global_entities = {'base': [], 'obstacle': []}

    with open(csv_path, 'r', newline='') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # Skip header lines (some logs may repeat headers)
            if line.startswith('rollout,step,entity_type,entity_id'):
                continue

            # Split into at most 5 fields: first 4 columns + remainder (x or weights string)
            parts = line.split(',', 4)
            if len(parts) < 5:
                continue

            rollout_s, step_s, etype, eid_s, rest = parts
            try:
                rollout = int(rollout_s)
                step = int(step_s)
                entity_id = int(eid_s)
            except ValueError:
                continue

            # For global base/obstacle, collect regardless of rollout filter
            if etype in ('base', 'obstacle'):
                xy = rest.split(',')
                if len(xy) == 2:
                    try:
                        x = float(xy[0]); y = float(xy[1])
                        global_entities[etype].append((x, y))
                    except ValueError:
                        pass

            # Apply rollout selection for per-rollout plots
            if r_start is not None and rollout < r_start:
                continue
            if r_end is not None and rollout > r_end:
                continue

            rd = rollouts.setdefault(rollout, RolloutData())

            if etype == 'weights':
                weights = safe_parse_weights(rest)
                if weights is not None:
                    rd.weights[entity_id] = weights
                continue

            # Numeric x,y rows
            xy = rest.split(',')
            if len(xy) != 2:
                continue
            try:
                x = float(xy[0])
                y = float(xy[1])
            except ValueError:
                continue

            ag, traj = parse_agent_traj(etype)
            if ag is not None:
                rd.trajs[(ag, traj)].append((step, x, y))
            else:
                rd.entities[etype].append((x, y))

    # Ensure trajectory points are time-ordered
    for rd in rollouts.values():
        for key in rd.trajs:
            rd.trajs[key].sort(key=lambda t: t[0])

    # Deduplicate global base/obstacle points while preserving order
    for k in ('base', 'obstacle'):
        seen = set()
        uniq = []
        for x, y in global_entities[k]:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                uniq.append((x, y))
        global_entities[k] = uniq

    return rollouts, global_entities

# ---- Plotting ---------------------------------------------------------------

def assign_agent_colors(rd: RolloutData):
    """Assign a unique pastel color to each agent present in this rollout (by trajs or weights)."""
    # Collect agent ids from trajectories and weights
    ag_set = set()
    for (ag, _traj) in rd.trajs.keys():
        ag_set.add(ag)
    for ag in rd.weights.keys():
        ag_set.add(ag)
    agents = sorted(ag_set)
    palette = sns.color_palette("pastel", max(3, len(agents)))  # ensure at least a few distinct colors
    return {ag: palette[i % len(palette)] for i, ag in enumerate(agents)}

def plot_map(rollout_idx: int, rd: RolloutData, outdir: str, dpi: int = 150,
             agent_colors=None, global_entities=None, traj_stride: int = 1):
    """
    Map for one rollout: entities + agent trajectories.
    Always overlays global base and obstacle markers (if any found).
    Agents are color-coded, and a marker is placed at the end of each agent path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # First, draw global base/obstacle so they appear beneath trajectories (sizes doubled)
    if global_entities:
        for etype in ('base', 'obstacle'):
            points = global_entities.get(etype, [])
            if points:
                xs, ys = zip(*points)
                marker = ENTITY_MARKERS.get(etype, 'o')
                if etype in 'base':
                    label = "Mothership"
                    color = 'blue'
                else:
                    label = "Obstacle"
                    color = 'black'
                ax.scatter(xs, ys, label=label, marker=marker, color=color, s=140, edgecolors='black', linewidths=0.7, alpha=0.8)

    # Then draw rollout-specific entities (e.g., tasks) (sizes doubled)
    for etype, points in rd.entities.items():
        if not points:
            continue
        # skip base/obstacle here since we already drew the global set
        if etype in ('base', 'obstacle'):
            continue
        xs, ys = zip(*points)
        marker = ENTITY_MARKERS.get(etype, 'o')
        if etype in 'task':
            label = "Task"
            color = 'green'
        ax.scatter(xs, ys, label=label, marker=marker, color=color, s=120, edgecolors='black', linewidths=0.5)

    # Plot trajectories: colored per agent
    by_agent = defaultdict(list)
    for (ag, traj), seq in rd.trajs.items():
        by_agent[ag].append(seq)

    for ag, seq_list in sorted(by_agent.items()):
        color = agent_colors.get(ag) if agent_colors else None
        last_point = None
        for seq in seq_list:
            if not seq:
                continue
            # Subsample waypoints for plotting
            seq_plot = seq[::max(1, traj_stride)]
            xs = [p[1] for p in seq_plot]
            ys = [p[2] for p in seq_plot]
            ax.plot(xs, ys, linewidth=2.5, alpha=0.95, color=color)
            # Keep true end of path for the agent marker
            first_point = (seq[0][1], seq[0][2])
        if first_point is not None:
            ax.scatter([first_point[0]], [first_point[1]], s=200, marker='o',
                    facecolor=color, edgecolor='black', linewidths=0.8, zorder=5,
                    label=f'Passenger {ag}')

    # ax.set_title(f"Rollout {rollout_idx}", fontsize=20)
    # ax.set_xlabel("x", fontsize=18)
    # ax.set_ylabel("y", fontsize=18)
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)

    # Build a non-duplicating legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        ax.legend(*zip(*uniq), loc='upper left', fontsize=12)

    fig.tight_layout()
    outpath = os.path.join(outdir, f"map_{rollout_idx}.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath

def plot_weights_multi_axes(rollout_idx: int, rd: RolloutData, outdir: str, dpi: int = 150, agent_colors=None):
    """
    Create a single figure with one subplot per agent showing that agent's weights.
    Y-axis fixed to [-1, 1]. Bars colored to match that agent's map color.
    """
    if not rd.weights:
        return None

    agents = sorted(rd.weights.keys())
    num_agents = len(agents)
    # Determine maximum length across agents for consistent x scale
    max_len = max(len(rd.weights[a]) for a in agents if isinstance(rd.weights[a], (list, tuple)))
    x = np.arange(max_len)

    # x_labels = ["Task", "Agent", "Frontier", "Near\nComm", "Needy\nComm", "Mother"]
    x_labels = [i+1 for i in range(max_len)]

    # Layout: one row, num_agents columns; widen figure as needed
    fig_w = max(8, 3 * num_agents)
    fig, axes = plt.subplots(1, num_agents, figsize=(fig_w, 4), sharey=True)
    if num_agents == 1:
        axes = [axes]

    for ax, ag in zip(axes, agents):
        w = rd.weights.get(ag, [])
        w_arr = np.array([float(v) for v in w] + [np.nan] * (max_len - len(w)))
        color = None
        if agent_colors and ag in agent_colors:
            color = agent_colors[ag]
        ax.bar(x, w_arr, width=0.8, color=color)
        ax.set_title(f'Passenger {ag}', fontsize=20)
        ax.set_xlabel('Specialization Idx', fontsize=16)
        ax.set_ylim([-1.0, 1.0])
        ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=14, rotation=0)

    axes[0].set_ylabel('Spec Value', fontsize=18)
    fig.suptitle(f"Rollout {rollout_idx} — Specializations by Passenger", fontsize=20)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    outpath = os.path.join(outdir, f"weights_{rollout_idx}.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV rollout log')
    parser.add_argument('--rollouts', type=int, nargs=2, metavar=('START', 'END'),
                        help='Inclusive range of rollout indices to plot')
    parser.add_argument('--outdir', type=str, default='figs', help='Output directory')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    parser.add_argument('--show', action='store_true', help='Show figures after saving')
    parser.add_argument('--traj-stride', type=int, default=1,
                    help='Plot every Nth waypoint for agent trajectories (>=1).')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    r_start = args.rollouts[0] if args.rollouts else None
    r_end = args.rollouts[1] if args.rollouts else None

    data, global_entities = load_log(args.csv, r_start, r_end)
    print(f"Loaded {len(data)} rollouts from {args.csv}")

    for r, rd in data.items():
        # Assign consistent colors for agents in this rollout
        agent_colors = assign_agent_colors(rd)
        print("Plotting map")
        map_path = plot_map(
                                r, rd, args.outdir, dpi=args.dpi,
                                agent_colors=agent_colors,
                                global_entities=global_entities,
                                traj_stride=args.traj_stride
                            )
        print('Plotting weights')
        weights_path = plot_weights_multi_axes(r, rd, args.outdir, dpi=args.dpi, agent_colors=agent_colors)

        if args.show:
            import matplotlib.image as mpimg
            for p in [map_path, weights_path]:
                if p and os.path.exists(p):
                    img = mpimg.imread(p)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(os.path.basename(p))
                    plt.show()

if __name__ == '__main__':
    main()
