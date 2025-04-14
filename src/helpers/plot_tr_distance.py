import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/Web02_run2_distance.csv")

if "TR" not in df.columns or "Distance" not in df.columns:
    raise ValueError("CSV must contain 'TR' and 'Distance' columns.")

plt.figure(figsize=(12, 6))
plt.plot(df["TR"], df["Distance"], label="Distance Traveled", linewidth=2)
plt.xlabel("TR")
plt.ylabel("Distance")
plt.title("Distance Traveled Per TR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()