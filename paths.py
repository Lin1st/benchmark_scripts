# paths.py
from pathlib import Path
import os


ROOT_DIR = Path(__file__).parent.resolve()

# Paths to data files
THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV = ROOT_DIR / "throughput_and_latency_vs_payload_size_under_load.csv"
BASELINE_LATENCY_CSV = ROOT_DIR / "baseline_latency.csv"
RESOURCE_VS_PAYLOAD_SIZE_CSV = ROOT_DIR / "resource_vs_payload_size.csv"
RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV = ROOT_DIR / "resource_and_latency_vs_request_rate.csv"
HEY_DEBUG_LOG = ROOT_DIR / "hey_debug_log.txt"
VEGETA_DEBUG_LOG = ROOT_DIR / "vegeta_debug_log.txt"

# Output directory for plots
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Output image files
OUTPUT_IMG_THROUGHPUT_VS_PAYLOAD_SIZE_UNDER_LOAD = PLOTS_DIR / "throughput_vs_payload_size_under_load.png"
OUTPUT_IMG_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD = PLOTS_DIR / "latency_vs_payload_size_under_load.png"
OUTPUT_IMG_CPU_USAGE_VS_PAYLOAD_SIZE = PLOTS_DIR / "cpu_usage_vs_payload_size.png"
OUTPUT_IMG_MEM_USAGE_VS_PAYLOAD_SIZE = PLOTS_DIR / "mem_usage_vs_payload_size.png"
OUTPUT_IMG_BASELINE_LATENCY = PLOTS_DIR / "baseline_latency.png"
OUTPUT_IMG_COMBINED_LATENCY = PLOTS_DIR / "combined_latency.png"
OUTPUT_IMG_CPU_USAGE_VS_REQUEST_RATE = PLOTS_DIR / "cpu_usage_vs_request_rate.png"
OUTPUT_IMG_MEM_USAGE_VS_REQUEST_RATE = PLOTS_DIR / "mem_usage_vs_request_rate.png"
OUTPUT_IMG_LATENCY_VS_REQUEST_RATE = PLOTS_DIR / "latency_vs_request_rate.png"

#  External benchmark directory template
#BENCH_DIR_TEMPLATE = str(Path.home() / "Desktop" / "bench_{}iB")
user_home = Path(os.path.expanduser(f"~{os.getenv('SUDO_USER', os.getenv('USER'))}"))
BENCH_DIR_TEMPLATE = str(user_home / "Desktop" / "bench_{}iB")

# External binary paths
WASMCLOUD_NATS = Path("/home/lin/Desktop/wasmCloud_nats/wasmCloud_first_try/target/release/wasmcloud")
WASMCLOUD_BYPASS = Path("/home/lin/Desktop/wasmCloud_mpsc/wasmCloud_first_try/target/release/wasmcloud")
WASH = Path("/home/lin/Desktop/wash")

# Provider reference
HTTP_PROVIDER_REFERENCE = "ghcr.io/wasmcloud/http-server:0.22.0"

# URL
URL = "http://127.0.0.1:8000"

