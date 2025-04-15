# run_test.py
import subprocess
import time
import csv
import os
from pathlib import Path

# Configurable paths
WASM_CLOUD_NATS = "/home/lin/Desktop/wasmCloud_nats/wasmCloud_first_try/target/release/wasmcloud"
WASM_CLOUD_BYPASS = "/home/lin/Desktop/wasmCloud_mpsc/wasmCloud_first_try/target/release/wasmcloud"
#WASM_CLOUD_BYPASS = "/home/lin/Desktop/wasmCloud_ds_1024/wasmCloud_first_try/target/release/wasmcloud"

WASH = "/home/lin/Desktop/wash"
BENCH_DIR_TEMPLATE = str(Path.home() / "Desktop" / "bench_{}B")

# Benchmark settings
PAYLOAD_SIZES = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RUNS_PER_SIZE = 20
HEY_DURATION = "5s"
HEY_CONCURRENCY = 10 
OUTPUT_CSV = "benchmark_results.csv"


def run_hey():
    result = subprocess.run(
        ["hey", "-z", HEY_DURATION, "-c", str(HEY_CONCURRENCY), "http://localhost:8000"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Hey failed:", result.stderr)
        return None
    return parse_hey_output(result.stdout)


def parse_hey_output(output):
    lines = output.splitlines()
    rps = latency = None
    for line in lines:
        if line.startswith("  Requests/sec"):
            rps = float(line.split()[1])
        if line.startswith("  Average"):
            latency = float(line.split()[1]) * 1000  # convert to ms
    return rps, latency


def write_result(scenario, payload_size, run_id, rps, latency):
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([scenario, payload_size, run_id, rps, latency])


def stop_all(scenario):
    # Stop components based on scenario
    aliases = ["http2"]
    if scenario == "composed":
        aliases = ["composed"]
    elif scenario in ["nats", "bypass"]:
        aliases.append("pong")

    for alias in aliases:
        subprocess.run([WASH, "stop", "component", alias], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([WASH, "stop", "provider", "http-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([WASH, "link", "del", "--all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(["pkill", "-f", "wasmcloud"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def run_scenario(scenario, wasmcloud_bin):
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, RUNS_PER_SIZE + 1):
            print(f"Running scenario={scenario} size={size} run={run_id}")
            stop_all(scenario)
            wasmcloud_proc = subprocess.Popen([
                "env",
                "WASMCLOUD_ALLOW_FILE_LOAD=true",
                "WASMCLOUD_RPC_HOST=127.0.0.1",
                "WASMCLOUD_CTL_HOST=127.0.0.1",
                wasmcloud_bin,
                "--max-components",
                "10"
            ])
            time.sleep(5)

            subprocess.run([WASH, "start", "provider", "ghcr.io/wasmcloud/http-server:0.22.0", "http-server"])

            if scenario == "composed":
                subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "composed", "wasi", "http"])
                subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/composed.wasm", "composed"])
            else:
                subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "http2", "wasi", "http"])
                subprocess.run([WASH, "link", "put", "--interface", "pingpong", "http2", "pong", "example", "pong"])

                subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/http-hello2/build/http_hello_world_s.wasm", "http2"])
                subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/pong/build/pong_s.wasm", "pong"])

            subprocess.run(["curl", "http://localhost:8000"])
            time.sleep(2)

            result = run_hey()
            if result:
                rps, latency = result
                write_result(scenario, size, run_id, rps, latency)

            stop_all(scenario)


if __name__ == "__main__":
    # Clear existing CSV
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "payload_size", "run_id", "requests_per_sec", "avg_latency_ms"])

    run_scenario("bypass", WASM_CLOUD_BYPASS)
    run_scenario("nats", WASM_CLOUD_NATS)
    run_scenario("composed", WASM_CLOUD_NATS)
