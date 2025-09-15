import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot average sum of rewards per run by model config.")
    parser.add_argument("csv_filepath", help="Path to the CSV file containing results.")
    parser.add_argument("--title", default="Average Task Data Value Returned per Run by Model Config", help="Plot title.")
    parser.add_argument("--xlabel", default="Model Config", help="X-axis label.")
    parser.add_argument("--ylabel", default="Average Task Data Value Returned", help="Y-axis label.")
    parser.add_argument("--ignore_0", action="store_true", help="Omit datapoints in step 0 for every run.")
    parser.add_argument("--model_config_order", nargs='+', help="Ordered list of model_configs to display in the plot.")
    parser.add_argument("--bar_labels", nargs='+', help="Custom labels for the bars, corresponding to model_config_order.")
    parser.add_argument("--save_stats", action="store_true", help="Save the mean and std stats to a LaTeX table file.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_filepath)
    if args.ignore_0 and 'step' in df.columns:
        df = df[df['step'] != 0]

    fig_name = os.path.splitext(args.csv_filepath)[0] + "_avg_rew_sums.png"
    sum_per_run = df.groupby(['model_config', 'run'])['reward'].sum().reset_index()
    grouped_mean = sum_per_run.groupby('model_config')['reward'].agg(['mean', 'std']).reset_index()

    if args.model_config_order:
        grouped_mean['model_config'] = pd.Categorical(grouped_mean['model_config'], categories=args.model_config_order, ordered=True)
        grouped_mean = grouped_mean.sort_values('model_config').dropna(subset=['model_config'])

    if args.bar_labels and len(args.bar_labels) != len(grouped_mean):
        raise ValueError(f"The number of bar_labels ({len(args.bar_labels)}) must match the number of model_configs ({len(grouped_mean)}).")

    print("Results:", grouped_mean)

    if args.save_stats:
        stats_df = grouped_mean.copy()
        if args.bar_labels:
            stats_df['model_config'] = args.bar_labels
        
        stats_df = stats_df.rename(columns={'model_config': 'Model Config', 'mean': 'Mean', 'std': 'Std. Dev.'})
        latex_filename = os.path.splitext(args.csv_filepath)[0] + "_stats.tex"
        stats_df.to_latex(latex_filename, index=False, float_format="%.2f", caption="Mean and Standard Deviation of Rewards by Model Configuration.", label="tab:stats")
        print(f"Stats saved to {latex_filename}")

    plt.figure(figsize=(8, 6))
    bar_positions = range(len(grouped_mean))
    plt.bar(bar_positions, grouped_mean['mean'], yerr=grouped_mean['std'], capsize=5)
    
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    
    labels = args.bar_labels if args.bar_labels else grouped_mean['model_config']
    plt.xticks(bar_positions, labels) #, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

if __name__ == "__main__":
    main()