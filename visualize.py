import argparse
import pandas as pd
import matplotlib.pyplot as plt

# python visualize.py --filepath eval_test\gif\render_0_actions.csv --step 0 --batch 0 --robot 0    

def plot_action_from_csv(filepath, step, batch, robot, color=None):
    # Load CSV
    df = pd.read_csv(filepath)

    # Filter for matching row
    row = df[(df["step"] == step) & (df["batch"] == batch) & (df["robot"] == robot)]

    if row.empty:
        print(f"No entry found for step={step}, batch={batch}, robot={robot}")
        return

    # Extract feature columns (assuming they start with 'f')
    f_values = row.filter(regex="^f").values.flatten()

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(f_values)), f_values, color=color)
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.title(f'Actions at Step {step}, Batch {batch}, Robot {robot}')
    plt.grid(True)
    plt.tight_layout()

    # Save figure at filepath with name containing batch, step, and robot
    out_path = f"{filepath}_batch{batch}__step{step}_robot{robot}_actions.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()

def plot_actions_step(filepath, step, batch, colors):
    # Plot all actions at this step and batch for all robots

    # Load CSV
    df = pd.read_csv(filepath)

    # Filter for matching step and batch
    filtered = df[(df["step"] == step) & (df["batch"] == batch)]

    if filtered.empty:
        print(f"No entries found for step={step}, batch={batch}")
        return

    # Get unique robots
    robots = filtered["robot"].unique()
    f_columns = filtered.filter(regex="^f").columns

    # Plot all robots' actions at this step as grid subplots, with each robot in a separate subplot
    n_robots = len(robots)
    n_cols = min(5, n_robots)
    n_rows = (n_robots + n_cols - 1) // n_cols


    print("Color:", colors)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, robot in enumerate(sorted(robots)):
        ax = axes[idx // n_cols][idx % n_cols]
        row = filtered[filtered["robot"] == robot]
        f_values = row[f_columns].values.flatten()
        color = colors[idx % len(colors)] if colors else None
        ax.bar(range(len(f_values)), f_values, color=color)
        ax.set_title(f"Robot {robot}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Hide unused subplots
    for idx in range(n_robots, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    plt.suptitle(f"Actions at Step {step}, Batch {batch} (All Robots)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save fig at filepath with name containing step and batch
    out_path = f"{filepath}_batch{batch}_step{step}_allrobots_actions.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()

def plot_actions_full_run(filepath, batch, robot, color=None):
    # Plot all actions over a run for robot
    # Load CSV
    df = pd.read_csv(filepath)

    # Filter for matching batch and robot
    filtered = df[(df["batch"] == batch) & (df["robot"] == robot)]

    if filtered.empty:
        print(f"No entries found for batch={batch}, robot={robot}")
        return

    # Extract feature columns for each step
    steps = filtered["step"].unique()
    f_columns = filtered.filter(regex="^f").columns

    # Plot each step's actions in a single figure, with grid subplots
    n_steps = len(steps)
    n_cols = min(5, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, step in enumerate(sorted(steps)):
        ax = axes[idx // n_cols][idx % n_cols]
        row = filtered[filtered["step"] == step]
        f_values = row[f_columns].values.flatten()
        ax.bar(range(len(f_values)), f_values, color=color)
        ax.set_title(f"Step {step}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Hide unused subplots
    for idx in range(n_steps, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    plt.suptitle(f"Actions for Batch {batch}, Robot {robot}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure at filepath with name containing batch and robot
    out_path = f"{filepath}_batch{batch}_robot{robot}_actions.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot action vector from saved CSV.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to actions CSV file")
    parser.add_argument("--step", type=int, default=None, help="Step index")
    parser.add_argument("--batch", type=int, required=True, help="Batch index")
    parser.add_argument("--robot", type=int, required=None, help="Robot index")
    parser.add_argument("--color", type=str, nargs="+", default=None, help="Bar color as list")

    args = parser.parse_args()
    if len(args.color) > 1:
        plot_actions_step(args.filepath, args.step, args.batch, args.color)
    elif args.step is not None:
        plot_action_from_csv(args.filepath, args.step, args.batch, args.robot, args.color[0])
    else:
        plot_actions_full_run(args.filepath, args.batch, args.robot, args.color[0])