# Re-import required libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparisons")
    parser.add_argument("--filepath", type=str, required=True, help="Path csv data file")

    args = parser.parse_args()

    # Load the most recent results file
    df_final = pd.read_csv(args.filepath) # "test_long_range_only/data/results.csv"

    # Define method columns
    methods = {
        "Policy": ("policy_reward_sum", "policy_reward_mean"),
        "Tasks Only": ("h_tasks_reward_sum", "h_tasks_reward_mean"),
        "Tasks+Comms": ("h_split_reward_sum", "h_split_reward_mean"),
        # "HybridDec": ("planner_reward_sum", "planner_reward_mean")
    }

    # Initialize data containers
    labels = []
    sum_means = []
    sum_stds = []
    mean_means = []
    mean_stds = []

    # Collect mean and std data
    for label, (sum_col, mean_col) in methods.items():
        if sum_col in df_final.columns and mean_col in df_final.columns:
            labels.append(label)
            sum_means.append(df_final[sum_col].mean())
            sum_stds.append(df_final[sum_col].std())
            mean_means.append(df_final[mean_col].mean())
            mean_stds.append(df_final[mean_col].std())

    # Plot
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], sum_means, width, yerr=sum_stds, capsize=8, label='Sum')
    ax.bar([i + width/2 for i in x], mean_means, width, yerr=mean_stds, capsize=8, label='Mean')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Task Value Returned')
    ax.set_title('Task Value Returned to Mothership for Policy and Hybrid-Decentralized Methods')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
