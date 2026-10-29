import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Your raw log data copied exactly as a string
raw_data = """

| Epoch | Model | Train Loss | Train Dice | Val Loss | Val Dice |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 9 |  can |0.2709 | 0.9250 | 0.2642 | 0.9244 |
| 10 | can | 0.2687 | 0.9250 | 0.2624 | 0.9270 |
| 11 | can | 0.2667 | 0.9292 | 0.2614 | 0.9274 |
| 12 | can | 0.2654 | 0.9337 | 0.2596 | 0.9298 |
| 1  | can | 0.5350 | 0.8334 | 0.3448 | 0.8319 |
| 2  | can | 0.3251 | 0.8804 | 0.3051 | 0.8858 |
| 3  | can | 0.3009 | 0.9033 | 0.2889 | 0.8967 |
| 4  | can | 0.2897 | 0.9104 | 0.2839 | 0.9027 |
| 5  | can | 0.2843 | 0.9046 | 0.2788 | 0.9079 |
| 6  | can | 0.2832 | 0.9025 | 0.2837 | 0.9023 |
| 7  | can | nan | 0.0505 | nan | 0.0524 |
| 8  | can | nan | 0.0523 | nan | 0.0524 |
| 9  | can | nan | 0.0510 | nan | 0.0524 |
| 10 | can | nan | 0.0508 | nan | 0.0524 |
| 1  | can | 0.5070 | 0.8336 | 0.3448 | 0.8343 |
| 2  | can | 0.3306 | 0.8817 | 0.3106 | 0.8760 |
| 3  | can | 0.3069 | 0.8939 | 0.2987 | 0.8828 |
| 4  | can | 0.2953 | 0.8995 | 0.2860 | 0.8984 |
| 5  | can | 0.2892 | 0.9082 | 0.2808 | 0.9048 |
| 6  | can | 0.2844 | 0.9108 | 0.2787 | 0.9088 |
| 7  | can | 0.2812 | 0.9059 | 0.2771 | 0.9084 |
| 8  | can | 0.2777 | 0.9164 | 0.2714 | 0.9169 |
| 9  | can | 0.2751 | 0.9250 | 0.2692 | 0.9192 |
| 10 | can | 0.2730 | 0.9171 | 0.2702 | 0.9166 |
| 11 | can | 0.2706 | 0.9174 | 0.2696 | 0.9166 |
| 12 | can | 0.2686 | 0.9288 | 0.2635 | 0.9254 |
| 1  | can | 0.4943 | 0.8364 | 0.3377 | 0.8390 |
| 2  | can | 0.3195 | 0.8900 | 0.2973 | 0.8878 |
| 3  | can | 0.2963 | 0.9042 | 0.2870 | 0.8971 |
| 4  | can | 0.2881 | 0.9071 | 0.2799 | 0.9086 |
| 5  | can | 0.2828 | 0.9034 | 0.2763 | 0.9133 |
| 6  | can | 0.2791 | 0.9137 | 0.2733 | 0.9162 |
| 7  | can | 0.2768 | 0.9261 | 0.2709 | 0.9180 |
| 8  | can | 0.2737 | 0.9196 | 0.2680 | 0.9224 |
| 9  | can | 0.2721 | 0.9212 | 0.2672 | 0.9218 |
| 10 | can | 0.2702 | 0.9264 | 0.2657 | 0.9254 |
| 11 | can | 0.2690 | 0.9329 | 0.2639 | 0.9268 |
| 12 | can | 0.2679 | 0.9317 | 0.2638 | 0.9241 |
| 13 | can | 0.2664 | 0.9261 | 0.2613 | 0.9284 |
| 14 | can | 0.2664 | 0.9284 | 0.2612 | 0.9310 |
| 15 | can | 0.2680 | 0.9263 | 0.2677 | 0.9264 |
| 16 | can | 1.3043 | 0.2098 | nan | 0.2097 |
| 17 | can | 0.0000 | 0.2099 | nan | 0.2097 |
| 18 | can | 0.0000 | 0.2099 | nan | 0.2097 |
| 19 | can | 0.0000 | 0.2101 | nan | 0.2097 |
| 20 | can | 0.0000 | 0.2098 | nan | 0.2097 |
| 1  | unet | 0.6259 | 0.7573 | 0.3872 | 0.7478 |
| 2  | unet | 0.3712 | 0.8525 | 0.3448 | 0.8487 |
| 3  | unet | 0.3443 | 0.8752 | 0.3279 | 0.8587 |
| 4  | unet | 0.3341 | 0.8510 | 0.3248 | 0.8615 |
| 5  | unet | 0.3273 | 0.8701 | 0.3272 | 0.8573 |
| 6  | unet | 0.3229 | 0.8725 | 0.3117 | 0.8762 |
| 7  | unet | 0.3190 | 0.8755 | 0.3255 | 0.8607 |
| 8  | unet | 0.3146 | 0.8860 | 0.3051 | 0.8808 |
| 9  | unet | 0.3123 | 0.8889 | 0.3128 | 0.8700 |
| 10 | unet | 0.3107 | 0.8788 | 0.2993 | 0.8872 |
| 11 | unet | 0.3087 | 0.8919 | 0.2966 | 0.8898 |
| 12 | unet | 0.3063 | 0.8943 | 0.2953 | 0.8887 |
| 13 | unet | 0.3033 | 0.8786 | 0.2990 | 0.8808 |
| 14 | unet | 0.3007 | 0.8662 | 0.3135 | 0.8597 |
| 15 | unet | 0.3027 | 0.8690 | 0.3029 | 0.8655 |
| 16 | unet | 0.2951 | 0.8909 | 0.2890 | 0.8924 |
| 17 | unet | 0.2950 | 0.8879 | 0.2909 | 0.8919 |
| 18 | unet | 0.2940 | 0.8962 | 0.2880 | 0.8940 |
| 19 | unet | 0.2929 | 0.9066 | 0.2865 | 0.8983 |
| 20 | unet | 0.2923 | 0.8801 | 0.2893 | 0.8919 |
| 21 | unet | 0.2914 | 0.8848 | 0.2958 | 0.8816 |
| 22 | unet | 0.2907 | 0.8995 | 0.2841 | 0.9014 |
| 23 | unet | 0.2902 | 0.9061 | 0.2909 | 0.8940 |
| 24 | unet | 0.2890 | 0.9013 | 0.2845 | 0.9001 |
| 25 | unet | 0.2893 | 0.8934 | 0.2828 | 0.9001 |
| 26 | unet | 0.2887 | 0.8987 | 0.2856 | 0.8958 |
| 1  | unet | 0.6825 | 0.6588 | 0.4000 | 0.6620 |
| 2  | unet | 0.3842 | 0.8279 | 0.3533 | 0.8203 |
| 3  | unet | 0.3499 | 0.8525 | 0.3288 | 0.8464 |
| 4  | unet | 0.3304 | 0.8642 | 0.3149 | 0.8681 |
| 5  | unet | 0.3223 | 0.8762 | 0.3082 | 0.8769 |
| 6  | unet | 0.3152 | 0.8795 | 0.3034 | 0.8811 |
| 7  | unet | 0.3106 | 0.8808 | 0.2988 | 0.8861 |
| 8  | unet | 0.3094 | 0.8734 | 0.2986 | 0.8851 |
| 9  | unet | 0.3080 | 0.8922 | 0.2960 | 0.8885 |
| 10 | unet | 0.3054 | 0.8928 | 0.3048 | 0.8807 |
"""

# 2. Parse Markdown text dynamically into a Pandas DataFrame
lines = [
    line.strip()
    for line in raw_data.strip().split("\n")
    if line.strip() and not line.strip().startswith("|:---")
]
columns = [col.strip() for col in lines[0].split("|")[1:-1]]
data_rows = [[cell.strip() for cell in line.split("|")[1:-1]] for line in lines[1:]]

df = pd.DataFrame(data_rows, columns=columns)

# Convert metric columns to numeric data types
df["Epoch"] = pd.to_numeric(df["Epoch"])
df["Val Dice"] = pd.to_numeric(df["Val Dice"])

# NEW: Drop any rows where 'Val Loss' or 'Train Loss' has 'nan' string value,
# and also filter out the collapsed low dice scores (< 0.3) that mess up the scaling.
df = df[~df["Val Loss"].str.lower().str.contains("nan")]
df = df[~df["Train Loss"].str.lower().str.contains("nan")]
df = df[df["Val Dice"] > 0.3]


# 3. Detect and separate independent training loops (runs)
run_labels = []
run_names = []
run_counter = 0
prev_epoch = -1
prev_model = ""
model_run_counters = {}

for idx, row in df.iterrows():
    epoch = row["Epoch"]
    model = row["Model"]

    # If the model switches or the epoch loops back to 1, mark a brand new run
    if model != prev_model or epoch <= prev_epoch:
        run_counter += 1
        model_run_counters[model] = model_run_counters.get(model, 0) + 1

    run_labels.append(run_counter)
    run_names.append(f"{model} (Run {model_run_counters[model]})")

    prev_epoch = epoch
    prev_model = model

df["Run_ID"] = run_labels
df["Run_Name"] = run_names


# 4. Generate the clean multi-line chart using Matplotlib
plt.figure(figsize=(10, 6))

# Unified palette themes for visual consistency
colors = {"can": "crimson", "unet": "dodgerblue"}

# Group the DataFrame by identified runs and map onto the axes
for (run_id, run_name), run_group in df.groupby(["Run_ID", "Run_Name"]):
    model_type = run_group["Model"].iloc[0]

    # Clean data sequence sorting by Epoch for correct line geometry
    run_sorted = run_group.sort_values(by="Epoch")

    plt.plot(
        run_sorted["Epoch"],
        run_sorted["Val Dice"],
        marker="o",
        linewidth=2,
        color=colors[model_type],
        label=run_name,
    )

# Formatting chart layout elements
plt.title(
    "Clean Multi-Run Validation Dice Comparison (NaN & Crashes Filtered)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation Dice Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Handle dynamic ticks range based on cleaned epoch counts
plt.xticks(range(1, df["Epoch"].max() + 1))
plt.ylim(0.6, 0.95)  # Zoom in onto the competitive space

plt.legend(title="Segmented Active Runs", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Display code output window
plt.show()
