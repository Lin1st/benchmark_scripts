# paths.py
from pathlib import Path


ROOT_DIR = Path(__file__).parent.resolve()

# Paths to data files
THROUGHPUT_CSV = ROOT_DIR / "benchmark_results.csv"
BASELINE_LATENCY_CSV = ROOT_DIR / "baseline_latency.csv"
RESOURCE_PROFILE_CSV = ROOT_DIR / "resource_profile.csv"
HEY_DEBUG_LOG = ROOT_DIR / "hey_debug_log.txt"

# Output directory for plots
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Output image files
OUTPUT_IMG_THROUGHPUT = PLOTS_DIR / "throughput_plot.png"
OUTPUT_IMG_LATENCY = PLOTS_DIR / "latency_plot.png"
OUTPUT_IMG_CPU = PLOTS_DIR / "cpu_plot.png"
OUTPUT_IMG_MEM = PLOTS_DIR / "memory_plot.png"
OUTPUT_IMG_BASELINE_LATENCY = PLOTS_DIR / "baseline_latency_plot.png"
OUTPUT_IMG_COMBINED_LATENCY = PLOTS_DIR / "combined_latency_plot.png"

#  External benchmark directory template
BENCH_DIR_TEMPLATE = str(Path.home() / "Desktop" / "bench_{}B")

# External binary paths
WASMCLOUD_NATS = Path("/home/lin/Desktop/wasmCloud_nats/wasmCloud_first_try/target/release/wasmcloud")
WASMCLOUD_BYPASS = Path("/home/lin/Desktop/wasmCloud_mpsc/wasmCloud_first_try/target/release/wasmcloud")
WASH = Path("/home/lin/Desktop/wash")

