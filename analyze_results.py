# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import t, norm
import os
from paths import (
    THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV,
    BASELINE_LATENCY_CSV,
    RESOURCE_VS_PAYLOAD_SIZE_CSV,
    RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV,
    OUTPUT_IMG_THROUGHPUT_VS_PAYLOAD_SIZE_UNDER_LOAD,
    OUTPUT_IMG_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD,
    OUTPUT_IMG_CPU_USAGE_VS_PAYLOAD_SIZE,
    OUTPUT_IMG_MEM_USAGE_VS_PAYLOAD_SIZE,
    OUTPUT_IMG_BASELINE_LATENCY,
    OUTPUT_IMG_COMBINED_LATENCY,
    OUTPUT_IMG_CPU_USAGE_VS_REQUEST_RATE,
    OUTPUT_IMG_MEM_USAGE_VS_REQUEST_RATE,
    OUTPUT_IMG_LATENCY_VS_REQUEST_RATE,
)


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


def summarize(df, value_col, conf_level, method, group_cols=None):
    # Define the correct order for payload sizes
    payload_order = ["0K", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M"]
    df["payload_size"] = pd.Categorical(df["payload_size"], categories=payload_order, ordered=True)

    if group_cols is None:
        group_cols = ["scenario", "payload_size"]

    summary = df.groupby(group_cols, observed=True).agg(
        mean_val=(value_col, "mean"),
        std_val=(value_col, "std"),
        count=(value_col, "count")
    ).reset_index()

    summary["ci"] = summary.apply(
        lambda row: compute_confidence_interval(row["std_val"], row["count"], conf_level, method),
        axis=1
    )
    return summary


def plot_with_ci_grouped(summary_df, group_cols, x_col, y_label, title, output_file, y_log_scale=False):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for group_keys, sub_df in summary_df.groupby(group_cols, observed=True):
        label = group_keys if isinstance(group_keys, str) else " | ".join(str(k) for k in group_keys)
        linestyle = "dashed" if "0b" in label.lower() or "baseline" in label.lower() else "solid"

        plt.errorbar(
            sub_df[x_col],
            sub_df["mean_val"],
            yerr=sub_df["ci"],
            label=label,
            capsize=5,
            marker="o",
            linestyle=linestyle
        )

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_label)
    plt.title(title)
    if y_log_scale:
        plt.yscale("log")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# ================== PLOTS ==================
# Throughput under load
def plot_throughput(df):
    throughput_summary = summarize(df, "throughput", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        throughput_summary,
        group_cols=["scenario"],
        x_col="payload_size",
        y_label="Throughput (requests/sec)",
        title="Throughput vs Payload Size",
        output_file=OUTPUT_IMG_THROUGHPUT_VS_PAYLOAD_SIZE_UNDER_LOAD
    )


# Latency (under stress)
def plot_latency_under_stress(df):
    latency_summary = summarize(df, "avg_latency_ms", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        latency_summary,
        group_cols=["scenario"],
        x_col="payload_size",
        y_label="Average Latency (ms)",
        title="Latency vs Payload Size",
        output_file=OUTPUT_IMG_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD
    )


# Latency: Baseline vs Under Stress
def plot_baseline_and_combined(baseline_df, stress_df):
    baseline_df["condition"] = "baseline"
    stress_df["condition"] = "under_stress"

    combined_df = pd.concat([
        stress_df[["scenario", "payload_size", "avg_latency_ms", "condition"]],
        baseline_df[["scenario", "payload_size", "avg_latency_ms", "condition"]]
    ])

    # Define the correct order for payload sizes
    payload_order = ["0K", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M"]
    combined_df["payload_size"] = pd.Categorical(combined_df["payload_size"], categories=payload_order, ordered=True)

    # Baseline Latency
    baseline_summary = summarize(baseline_df, "avg_latency_ms", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        baseline_summary,
        group_cols=["scenario"],
        x_col="payload_size",
        y_label="Baseline Latency (ms)",
        title="Baseline Latency vs Payload Size",
        output_file=OUTPUT_IMG_BASELINE_LATENCY
    )

    # Combined Latency (baseline + under stress)
    combined_summary = combined_df.groupby(["scenario", "payload_size", "condition"], observed=True).agg(
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
        x_col="payload_size",
        y_label="Latency (ms)",
        title="Latency: Baseline vs Under Stress",
        output_file=OUTPUT_IMG_COMBINED_LATENCY
    )


# CPU and Memory Usage
def plot_resource_usage(resource_df, include_pss=True):
    # Normalize column names to ensure case consistency
    resource_df.columns = resource_df.columns.str.strip().str.lower()

    required_columns = {"rss_mib", "cpu_percent"}
    if include_pss:
        required_columns.add("pss_mib")
    if not required_columns.issubset(resource_df.columns):
        print(f"Missing required columns for resource usage: {required_columns - set(resource_df.columns)}")
        return

    # Plot RSS (and optionally PSS)
    if include_pss:
        # Reshape the data to include a "metric" column for PSS and RSS
        memory_df = pd.melt(
            resource_df,
            id_vars=["scenario", "payload_size"],
            value_vars=["pss_mib", "rss_mib"],
            var_name="metric",
            value_name="memory_mib"
        )

        # Combine "scenario" and "metric" into a single group for plotting
        memory_df["scenario"] = memory_df["scenario"] + " | " + memory_df["metric"].str.upper()

        # Use the summarize function to compute statistics
        memory_summary = summarize(memory_df, "memory_mib", CONF_LEVEL, CI_METHOD)

        plot_with_ci_grouped(
            memory_summary,
            group_cols=["scenario"],
            x_col="payload_size",
            y_label="Memory Usage (MiB)",
            title="Memory Usage (PSS and RSS) vs Payload Size",
            output_file=OUTPUT_IMG_MEM_USAGE_VS_PAYLOAD_SIZE
        )
    else:
        # Plot only RSS
        rss_summary = summarize(resource_df, "rss_mib", CONF_LEVEL, CI_METHOD)
        plot_with_ci_grouped(
            rss_summary,
            group_cols=["scenario"],
            x_col="payload_size",
            y_label="Memory Usage (MiB)",
            title="Memory Usage (RSS) vs Payload Size",
            output_file=OUTPUT_IMG_MEM_USAGE_VS_PAYLOAD_SIZE
        )

    # Plot CPU usage
    cpu_summary = summarize(resource_df, "cpu_percent", CONF_LEVEL, CI_METHOD)
    plot_with_ci_grouped(
        cpu_summary,
        group_cols=["scenario"],
        x_col="payload_size",
        y_label="CPU Usage (%)",
        title="CPU Usage vs Payload Size",
        output_file=OUTPUT_IMG_CPU_USAGE_VS_PAYLOAD_SIZE
    )


def plot_resource_vs_request_rate(resource_vs_request_rate_df, include_pss=True):
    # Normalize column names to ensure case consistency
    resource_vs_request_rate_df.columns = resource_vs_request_rate_df.columns.str.strip().str.lower()

    # print("Normalized column names:", resource_vs_request_rate_df.columns.tolist())

    required_columns = {"rss_mib", "cpu_percent", "request_rate"}
    if include_pss:
        required_columns.add("pss_mib")
    if not required_columns.issubset(resource_vs_request_rate_df.columns):
        print(f"Missing required columns for resource usage: {required_columns - set(resource_vs_request_rate_df.columns)}")
        return

    # Plot CPU usage
    cpu_summary = summarize(resource_vs_request_rate_df, "cpu_percent", CONF_LEVEL, CI_METHOD, group_cols=["scenario", "payload_size", "request_rate"])
    plot_with_ci_grouped(
        cpu_summary,
        group_cols=["scenario", "payload_size"],
        x_col="request_rate",
        y_label="CPU Usage (%)",
        title="CPU Usage vs Request Rate",
        output_file=OUTPUT_IMG_CPU_USAGE_VS_REQUEST_RATE
    )

    # Plot memory usage (RSS and optionally PSS)
    if include_pss:
        for metric, y_label, output_file in [
            ("pss_mib", "PSS Memory Usage (MiB)", OUTPUT_IMG_MEM_USAGE_VS_REQUEST_RATE),
            ("rss_mib", "RSS Memory Usage (MiB)", OUTPUT_IMG_MEM_USAGE_VS_REQUEST_RATE)
        ]:
            memory_summary = summarize(resource_vs_request_rate_df, metric, CONF_LEVEL, CI_METHOD, group_cols=["scenario", "payload_size", "request_rate"])
            plot_with_ci_grouped(
                memory_summary,
                group_cols=["scenario", "payload_size"],
                x_col="request_rate",
                y_label=y_label,
                title=f"{y_label} vs Request Rate",
                output_file=output_file
            )
    else:
        # Plot only RSS
        rss_summary = summarize(resource_vs_request_rate_df, "rss_mib", CONF_LEVEL, CI_METHOD, group_cols=["scenario", "payload_size", "request_rate"])
        plot_with_ci_grouped(
            rss_summary,
            group_cols=["scenario", "payload_size"],
            x_col="request_rate",
            y_label="Memory Usage (MiB)",
            title="Memory Usage vs Request Rate",
            output_file=OUTPUT_IMG_MEM_USAGE_VS_REQUEST_RATE
        )


def plot_latency_vs_request_rate(resource_vs_request_rate_df):
    latency_summary = resource_vs_request_rate_df.groupby(
        ["scenario", "payload_size", "request_rate"], observed=True
    ).agg(
        mean_val=("avg_latency_ms", "mean"),
        std_val=("avg_latency_ms", "std"),
        count=("avg_latency_ms", "count")
    ).reset_index()

    latency_summary["ci"] = latency_summary.apply(
        lambda row: compute_confidence_interval(row["std_val"], row["count"], CONF_LEVEL, CI_METHOD),
        axis=1
    )

    plot_with_ci_grouped(
        latency_summary,
        group_cols=["scenario", "payload_size"],
        x_col="request_rate",
        y_label="Latency (ms)",
        title="Latency vs Request Rate",
        output_file=OUTPUT_IMG_LATENCY_VS_REQUEST_RATE,
        y_log_scale=True
    )


# ========================= MAIN =========================
def main():
    #df = pd.read_csv(THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV )
    #plot_throughput(df)
    #plot_latency_under_stress(df)

    if os.path.exists(BASELINE_LATENCY_CSV):
        baseline_df = pd.read_csv(BASELINE_LATENCY_CSV)
        plot_baseline_and_combined(baseline_df, df)
    else:
        print(f"Baseline latency file '{BASELINE_LATENCY_CSV}' not found.")
        
    if os.path.exists(RESOURCE_VS_PAYLOAD_SIZE_CSV):
        resource_df = pd.read_csv(RESOURCE_VS_PAYLOAD_SIZE_CSV)
        plot_resource_usage(resource_df)
    else:
        print(f"Resource usage file '{RESOURCE_VS_PAYLOAD_SIZE_CSV}' not found.")

    if os.path.exists(RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV):
        resource_vs_request_rate_df = pd.read_csv(RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV)
        plot_resource_vs_request_rate(resource_vs_request_rate_df)
        plot_latency_vs_request_rate(resource_vs_request_rate_df)
    else:
        print(f"Resource vs Request Rate file '{RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV}' not found.")


if __name__ == "__main__":
    main()
