# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import t, norm
import os

# ================== PATHS ==================
THROUGHPUT_CSV = "benchmark_results.csv"
RESOURCE_CSV = "resource_profile.csv"
BASELINE_LATENCY_CSV = "baseline_latency.csv"

OUTPUT_IMG_THROUGHPUT = "throughput_plot.png"
OUTPUT_IMG_LATENCY = "latency_plot.png"
OUTPUT_IMG_CPU = "cpu_plot.png"
OUTPUT_IMG_MEM = "memory_plot.png"
OUTPUT_IMG_BASELINE_LATENCY = "baseline_latency_plot.png"
OUTPUT_IMG_COMBINED_LATENCY = "combined_latency_plot.png"


# ================== ANALYSIS SETTINGS ==================
CONF_LEVEL = 0.95
CI_METHOD = "std"  # Options: "t", "z", "std"


# ================== UTILS ==================
def compute_confidence_interval(std, n, conf_level, method):
    if n <= 1 or pd.isna(std):
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


def summarize(df, value_col, conf_level, method):
    summary = df.groupby(["scenario", "payload_size"]).agg(
        mean_val=(value_col, "mean"),
        std_val=(value_col, "std"),
        count=(value_col, "count")
    ).reset_index()

    summary["ci"] = summary.apply(
        lambda row: compute_confidence_interval(row["std_val"], row["count"], conf_level, method),
        axis=1
    )
    return summary


def plot_with_ci_grouped(summary_df, group_cols, y_label, title, output_file, y_log_scale=False):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for group_keys, sub_df in summary_df.groupby(group_cols):
        label = group_keys if isinstance(group_keys, str) else " | ".join(str(k) for k in group_keys)
        linestyle = "dashed" if "baseline" in label.lower() else "solid"

        plt.errorbar(
            sub_df["payload_size"],
            sub_df["mean_val"],
            yerr=sub_df["ci"],
            label=label,
            capsize=5,
            marker="o",
            linestyle=linestyle
        )

    plt.xlabel("Payload Size (bytes)")
    plt.ylabel(y_label)
    plt.title(title)
    if y_log_scale:
        plt.yscale("log")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# ================== PLOTS ==================
# Throughput (under stress)
def plot_throughput(df):
    throughput_summary = summarize(df, "requests_per_sec", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        throughput_summary,
        group_cols=["scenario"],
        y_label="Throughput (requests/sec)",
        title="Throughput vs Payload Size",
        output_file=OUTPUT_IMG_THROUGHPUT
    )


# Latency (under stress)
def plot_latency_under_stress(df):
    latency_summary = summarize(df, "avg_latency_ms", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        latency_summary,
        group_cols=["scenario"],
        y_label="Average Latency (ms)",
        title="Latency vs Payload Size",
        output_file=OUTPUT_IMG_LATENCY
    )


# Latency: Baseline vs Under Stress
def plot_baseline_and_combined(baseline_df, stress_df):
    baseline_df["condition"] = "baseline"
    stress_df["condition"] = "under_stress"

    combined_df = pd.concat([
        stress_df[["scenario", "payload_size", "avg_latency_ms", "condition"]],
        baseline_df[["scenario", "payload_size", "avg_latency_ms", "condition"]]
    ])

    # Baseline Latency
    baseline_summary = summarize(baseline_df, "avg_latency_ms", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        baseline_summary,
        group_cols=["scenario"],
        y_label="Baseline Latency (ms)",
        title="Baseline Latency vs Payload Size",
        output_file=OUTPUT_IMG_BASELINE_LATENCY
    )

    # Combined Latency (baseline + under stress)
    combined_summary = combined_df.groupby(["scenario", "payload_size", "condition"]).agg(
        mean_val=("avg_latency_ms", "mean"),
        std_val=("avg_latency_ms", "std"),
        count=("avg_latency_ms", "count")
    ).reset_index()

    combined_summary["ci"] = combined_summary.apply(
        lambda row: compute_confidence_interval(row["std_val"], row["count"], CONF_LEVEL, CI_METHOD),
        axis=1
    )

    plot_with_ci_grouped(
        combined_summary,
        group_cols=["scenario", "condition"],
        y_label="Latency (ms)",
        title="Latency: Baseline vs Under Stress",
        output_file=OUTPUT_IMG_COMBINED_LATENCY
    )


# CPU and Memory Usage
def plot_resource_usage(resource_df):
    cpu_summary = summarize(resource_df, "cpu_percent", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        cpu_summary,
        group_cols=["scenario"],
        y_label="CPU Usage (%)",
        title="CPU Usage vs Payload Size",
        output_file=OUTPUT_IMG_CPU
    )

    mem_summary = summarize(resource_df, "memory_mb", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        mem_summary,
        group_cols=["scenario"],
        y_label="Memory Usage (MB)",
        title="Memory Usage vs Payload Size",
        output_file=OUTPUT_IMG_MEM
    )


# ========================= MAIN =========================
def main():
    df = pd.read_csv(THROUGHPUT_CSV)
    plot_throughput(df)
    plot_latency_under_stress(df)

    if os.path.exists(BASELINE_LATENCY_CSV):
        baseline_df = pd.read_csv(BASELINE_LATENCY_CSV)
        plot_baseline_and_combined(baseline_df, df)
    else:
        print(f"Baseline latency file '{BASELINE_LATENCY_CSV}' not found.")
        
    if os.path.exists(RESOURCE_CSV):
        resource_df = pd.read_csv(RESOURCE_CSV)
        plot_resource_usage(resource_df)
    else:
        print(f"Resource usage file '{RESOURCE_CSV}' not found.")


if __name__ == "__main__":
    main()
