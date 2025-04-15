# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import t, norm

# Configuration
CSV_FILE = "benchmark_results.csv"
OUTPUT_IMG_THROUGHPUT = "throughput_plot.png"
OUTPUT_IMG_LATENCY = "latency_plot.png"
CONF_LEVEL = 0.95  # for 95% confidence interval
CI_METHOD = "std"  # Options: "t", "z", or "std"

# Load data
df = pd.read_csv(CSV_FILE)

# Compute confidence interval
def compute_confidence_interval(std, n, conf_level, method):
    if n <= 1:
        return 0
    if method == "std":
        return std
    if method == "t":
        critical_value = t.ppf((1 + conf_level) / 2, df=n - 1)
    elif method == "z":
        critical_value = norm.ppf((1 + conf_level) / 2)
    else:
        raise ValueError("Unsupported CI method. Use 't', 'z', or 'std'.")
    return critical_value * std / (n ** 0.5)

# Summarize data
def summarize(data, value_col, conf_level, method):
    summary = data.groupby(["scenario", "payload_size"]).agg(
        mean_val=(value_col, "mean"),
        std_val=(value_col, "std"),
        count=(value_col, "count")
    ).reset_index()

    summary["ci"] = summary.apply(
        lambda row: compute_confidence_interval(row["std_val"], row["count"], conf_level, method),
        axis=1
    )

    return summary

# Plotting function
def plot_with_ci(summary_df, y_label, title, output_file, y_log_scale=False):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    linestyles = {
        "bypass": "dashed",
        "nats": "dotted",
        "composed": "solid"
    }

    for scenario in summary_df["scenario"].unique():
        scenario_df = summary_df[summary_df["scenario"] == scenario]
        plt.errorbar(
            scenario_df["payload_size"],
            scenario_df["mean_val"],
            yerr=scenario_df["ci"],
            label=scenario,
            capsize=5,
            marker='o',
            linestyle=linestyles.get(scenario, "solid")
        )

    plt.xlabel("Payload Size (bytes)")
    plt.ylabel(y_label)
    plt.title(title)
    if y_log_scale:
        plt.yscale("log")
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Throughput
throughput_summary = summarize(df, "requests_per_sec", CONF_LEVEL, CI_METHOD)
plot_with_ci(
    throughput_summary,
    y_label="Throughput (requests/sec)",
    title=f"Throughput vs Payload Size with {'Standard Deviation' if CI_METHOD == 'std' else f'{int(CONF_LEVEL * 100)}% Confidence Interval'}",
    output_file=OUTPUT_IMG_THROUGHPUT,
    y_log_scale=False
)

# Latency under stress
latency_summary = summarize(df, "avg_latency_ms", CONF_LEVEL, CI_METHOD)
plot_with_ci(
    latency_summary,
    y_label="Average Latency (ms)",
    title=f"Latency vs Payload Size with {'Standard Deviation' if CI_METHOD == 'std' else f'{int(CONF_LEVEL * 100)}% Confidence Interval'}",
    output_file=OUTPUT_IMG_LATENCY,
    y_log_scale=False
)

