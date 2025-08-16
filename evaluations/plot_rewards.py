import pandas as pd
import matplotlib.pyplot as plt

# Reload the latest uploaded CSV file after reset
df_newest = pd.read_csv("evaluations\\data\\results2.csv")

# Sum rewards within each run for each model_config
sum_per_run_newest = df_newest.groupby(['model_config', 'run'])['reward'].sum().reset_index()

# Compute mean and std of these sums for each model_config
grouped_sum_newest = sum_per_run_newest.groupby('model_config')['reward'].agg(['mean', 'std']).reset_index()

# Plot
plt.figure(figsize=(8, 6))
plt.bar(grouped_sum_newest['model_config'], grouped_sum_newest['mean'], yerr=grouped_sum_newest['std'], capsize=5)
plt.xlabel('Model Config')
plt.ylabel('Average Sum of Rewards')
plt.title('Average Sum of Rewards per Run by Model Config with Error Bars')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()