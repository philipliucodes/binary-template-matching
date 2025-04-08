import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ground_truth_df = pd.read_csv("data/csvs/Web47_1_mouse _stretched.csv")
recorded_df = pd.read_csv("output/Web47_run1_distance.csv")

merged_df = pd.merge(ground_truth_df, recorded_df, on="TR", suffixes=("_truth", "_recorded"))

correlation, p_value = pearsonr(merged_df["Distance_truth"], merged_df["Distance_recorded"])
print(f"Pearson Correlation: {correlation:.3f} (p-value: {p_value:.4f})")

plt.figure(figsize=(12, 6))
plt.plot(merged_df["TR"], merged_df["Distance_truth"], label="Ground Truth", linewidth=2)
plt.plot(merged_df["TR"], merged_df["Distance_recorded"], label="Recorded Data", linewidth=2, linestyle='--')
plt.xlabel("TR")
plt.ylabel("Distance")
plt.title("Distance Traveled Per TR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
