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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot action vector from saved CSV.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to actions CSV file")
    parser.add_argument("--step", type=int, required=True, help="Step index")
    parser.add_argument("--batch", type=int, required=True, help="Batch index")
    parser.add_argument("--robot", type=int, required=True, help="Robot index")
    parser.add_argument("--color", type=str, default=None, help="Bar color (e.g., 'blue', '#FF5733')")

    args = parser.parse_args()

    plot_action_from_csv(args.filepath, args.step, args.batch, args.robot, args.color)
