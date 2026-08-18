import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === STPLS3D per-class IoU ===
results_KPConv = {
    "equal":  [86.493, 80.128, 66.169, 49.832, 49.189, 10.748],
    "invf":   [85.371, 74.248, 64.457, 27.737, 14.798, 14.630],
    "cb":     [87.103, 81.918, 66.452, 51.929, 51.307, 10.458],
    "invl":   [87.052, 81.551, 66.655, 41.436, 66.402, 12.607],
    "invp":   [87.137, 79.778, 67.001, 42.094, 51.101, 11.468],
    "comf":   [87.047, 79.856, 66.938, 39.793, 53.022, 13.603],
    "FL":     [87.737, 80.737, 68.685, 45.781, 50.606, 11.138],
    "LDAM":   [87.317, 81.082, 68.016, 48.747, 50.894, 10.724],
    "LADJ":   [86.809, 81.431, 66.363, 40.872, 40.175, 12.614],
    "BS":     [85.546, 73.559, 65.934, 20.434, 6.612 , 12.844],
    "SS":     [85.399, 73.363, 66.313, 41.072, 52.228, 10.779]
}

results_RandLA = {
    "equal":  [85.007, 76.260, 66.441, 40.978, 44.774,  7.322],
    "invf":   [79.011, 64.866, 59.558, 29.967, 31.597,  8.769],
    "cb":     [82.205, 77.489, 62.999, 51.747, 27.566,  7.895],
    "invl":   [80.987, 73.378, 60.836, 46.349, 23.336,  6.267],
    "invp":   [82.773, 76.344, 63.670, 50.526, 53.845, 12.484],
    "comf":   [86.634, 77.786, 71.537, 46.661, 54.496, 11.472],
    "FL":     [84.037, 76.817, 65.055, 44.904, 19.221,  6.396],
    "LDAM":   [84.116, 80.019, 64.938, 45.839, 27.797, 10.906],
    "LADJ":   [83.011, 63.219, 66.775, 24.551,  8.356,  5.701],
    "BS":     [80.360, 65.047, 61.098, 38.773, 25.817,  3.906],
    "SS":     [82.069, 68.203, 62.073, 45.178, 31.936,  4.470]
}


# Select which results to visualize
data = results_RandLA

classes = ["ground", "building", "vegetation", "cars", "lightStreetSigns", "fences"]

df = pd.DataFrame(data, index=classes)

# Compute delta vs equal
delta_df = df.subtract(df["equal"], axis=0)

fig, ax = plt.subplots(figsize=(20,12))
width = 0.1
x = np.arange(len(classes)) * 1.3  # Increase spacing between class groups

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"]
methods = delta_df.columns.tolist()
methods.remove("equal")

for i, method in enumerate(methods):
    ax.bar(x + i*width, delta_df[method], width=width, label=method)

ax.set_xticks(x + width*len(methods)/2 - width/2)
ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=34)
ax.set_ylabel("Δ IoU vs uniform (%)", fontsize=38)
# Determine network name from selected data
network_name = "KPConv" if data is results_KPConv else "RandLA-Net"
ax.set_title(f"STPLS3D ({network_name}): Class-wise IoU delta relative to uniform", fontsize=38)
ax.axhline(0, color='black', linewidth=0.8)

for i in range(len(classes) - 1):
    ax.axvline(x=x[i] + 1.15, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax.legend(ncol=3, loc="lower left", fontsize=34)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.5)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()
ax.tick_params(axis='y', labelsize=34)

plt.tight_layout()

plt.savefig(f"STPLS3D_{network_name}_deltas.png", dpi=300, bbox_inches='tight', format='png')

# Maximise the window when running interactively; only supported on some
# backends (e.g. TkAgg), so ignore failures on headless/other backends
try:
    plt.get_current_fig_manager().window.state('zoomed')
except Exception:
    pass

plt.show()