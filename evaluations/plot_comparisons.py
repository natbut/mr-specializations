# Re-import required libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparisons")
    parser.add_argument("--filepath", type=str, required=True, help="Path csv data file")
    parser.add_argument("--fixed-obj", action='store_true')

    args = parser.parse_args()

    # Load the most recent results file
    df_final = pd.read_csv(args.filepath)  # "test_long_range_only/data/results.csv"

    # Define method columns
    if args.fixed_obj:
        methods = {
            "Tasks Only": ("tasks_reward_sum", "tasks_reward_mean"),
            "Tasks+Comms": ("taskcomms_reward_sum", "taskcomms_reward_mean"),
            "Tasks+Explore": ("taskexp_reward_sum", "taskexp_reward_mean"),
            "2T+1C+1E": ("taskcommsexpA_reward_sum", "taskcommsexpA_reward_mean"),
            "1T+2C+1E": ("taskcommsexpB_reward_sum", "taskcommsexpB_reward_mean"),
            "1T+1C+2E": ("taskcommsexpC_reward_sum", "taskcommsexpC_reward_mean"),
            "MCAPS": ("policy_reward_sum", "policy_reward_mean"),
        }
        title="Comparisons Against Fixed Objective Functions"
    else:
        methods = {
            "HybridDec": ("planner_reward_sum", "planner_reward_mean"),
            "Tasks Only": ("h_tasks_reward_sum", "h_tasks_reward_mean"),
            "Tasks+Comms": ("h_split_reward_sum", "h_split_reward_mean"),
            "MCAPS": ("policy_reward_sum", "policy_reward_mean"),
        }
        title="Comparisons"

    # Initialize data containers
    labels = []
    sum_means = []
    sum_sems = []  # use SEM instead of std
    mean_means = []
    mean_sems = []

    # Collect mean and SEM data
    for label, (sum_col, mean_col) in methods.items():
        if sum_col in df_final.columns and mean_col in df_final.columns:
            labels.append(label)
            n_sum = df_final[sum_col].count()
            n_mean = df_final[mean_col].count()

            sum_means.append(df_final[sum_col].mean())
            sum_sems.append(df_final[sum_col].std() / np.sqrt(n_sum))

            mean_means.append(df_final[mean_col].mean())
            mean_sems.append(df_final[mean_col].std() / np.sqrt(n_mean))

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

    fig, ax = plt.subplots(figsize=(17, 6.5))
    bars = ax.bar(
        x, sum_means, width,
        yerr=sum_sems, align='center', capsize=5,
        label='Sum', color=sns.color_palette()[0]
    )
    ax.bar_label(bars, label_type='edge', fmt='%.3f', fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Task Value Returned')
    ax.set_title(title, fontsize=32)
    plt.tight_layout()
    plt.show()
