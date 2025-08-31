# Re-import required libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparisons")
    parser.add_argument("--filepath", type=str, required=True, help="Path csv data file")

    args = parser.parse_args()

    # Load the most recent results file
    df_final = pd.read_csv(args.filepath) # "test_long_range_only/data/results.csv"

    # Define method columns
    methods = {
        "HybridDec": ("planner_reward_sum", "planner_reward_mean"),
        "Tasks Only": ("h_tasks_reward_sum", "h_tasks_reward_mean"),
        "Tasks+Comms": ("h_split_reward_sum", "h_split_reward_mean"),
        "MACPS": ("policy_reward_sum", "policy_reward_mean"),
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
    width = 0.5

    # Set seaborn color palette and style
    sns.set_palette("pastel")
    sns.set_style("whitegrid")

    # Increase font sizes globally
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(x, sum_means, width, yerr=sum_stds, align='center', capsize=5, label='Sum', color=sns.color_palette()[0])
    ax.bar_label(bars, label_type='edge', fmt='%.3f', fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Task Value Returned')
    ax.set_title('Long-Range Task Comparisons')
    plt.tight_layout()
    plt.show()
