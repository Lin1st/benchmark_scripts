# run_test.py
import subprocess
import time
import csv
import os
import psutil
import socket
import uuid
from pathlib import Path
from threading import Thread, Event as ThreadEvent


# Paths
WASM_CLOUD_NATS = "/home/lin/Desktop/wasmCloud_nats/wasmCloud_first_try/target/release/wasmcloud"
WASM_CLOUD_BYPASS = "/home/lin/Desktop/wasmCloud_mpsc/wasmCloud_first_try/target/release/wasmcloud"
WASH = "/home/lin/Desktop/wash"
BENCH_DIR_TEMPLATE = str(Path.home() / "Desktop" / "bench_{}B")

THROUGHPUT_CSV = "benchmark_results.csv"
BASELINE_LATENCY_CSV = "baseline_latency.csv"
RESOURCE_PROFILE_CSV = "resource_profile.csv"
HEY_DEBUG_LOG = "hey_debug_log.txt"

# Benchmark settings
PAYLOAD_SIZES = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RUNS_PER_SIZE = 5
BASELINE_LATENCY_RUNS = 5
HEY_DURATION = "1s"
HEY_CONCURRENCY = 3
RESOURCE_SAMPLE_INTERVAL = 0.5


# ================== UTILS ==================
def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not available after {timeout}s")


def build_systemd_cmd(suffix=None, limit_usage=False):
    unit_name = f"wasmbench-{suffix or uuid.uuid4().hex[:8]}"
    cmd = ["systemd-run", "--user", "--scope", f"--unit={unit_name}"]
    if limit_usage:
        cmd += ["-p", "CPUQuota=50%", "-p", "MemoryMax=512M"]
    return cmd


def write_result(path, headers, row):
    first_write = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(headers)
        writer.writerow(row)


def stop_all(scenario):
    aliases = {"bypass": ["http2", "pong"], "nats": ["http2", "pong"], "composed": ["composed"]}.get(scenario, [])
    for alias in aliases:
        subprocess.run([WASH, "stop", "component", alias], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([WASH, "stop", "provider", "http-server"], stdout=subprocess.DEVNULL)
    subprocess.run([WASH, "link", "del", "--all"], stdout=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "wasmcloud"], stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "-f", "nats-server"], stdout=subprocess.DEVNULL)
    time.sleep(2)


def start_wasmcloud(scenario, wasmcloud_bin, bench_path, run_id, limit_usage=False):
    subprocess.run(build_systemd_cmd(f"nats-{run_id}") + [
        "docker", "run", "-d", "--name", "nats-server",
        "-p", "4222:4222", "-p", "8222:8222", "nats:latest", "-js"
    ])
    wait_for_port("127.0.0.1", 4222)
    time.sleep(5)

    subprocess.Popen(build_systemd_cmd(f"host-{run_id}", limit_usage=limit_usage) + [
        "env",
        "WASMCLOUD_ALLOW_FILE_LOAD=true",
        "WASMCLOUD_RPC_HOST=127.0.0.1",
        "WASMCLOUD_CTL_HOST=127.0.0.1",
        wasmcloud_bin,
        "--max-components", "10"
    ])
    time.sleep(20)

    subprocess.run([WASH, "start", "provider", "ghcr.io/wasmcloud/http-server:0.22.0", "http-server"])
    time.sleep(2)

    if scenario == "composed":
        subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "composed", "wasi", "http"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/composed.wasm", "composed"])
    else:
        subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "http2", "wasi", "http"])
        subprocess.run([WASH, "link", "put", "--interface", "pingpong", "http2", "pong", "example", "pong"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/http-hello2/build/http_hello_world_s.wasm", "http2"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/pong/build/pong_s.wasm", "pong"])

    wait_for_port("127.0.0.1", 8000, timeout=15)

    for attempt in range(10):
        try:
            r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8000"],
                               capture_output=True, text=True)
            if r.stdout.strip() == "200":
                print("Warm-up HTTP request successful")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Warm-up HTTP request failed repeatedly (no 200 OK)")


def run_hey(scenario=None, payload_size=None, run_id=None, concurrency=HEY_CONCURRENCY, duration=HEY_DURATION):
    result = subprocess.run(
        ["hey", "-z", duration, "-c", str(concurrency), "http://localhost:8000"],
        capture_output=True,
        text=True,
    )
    with open(HEY_DEBUG_LOG, "a") as f:
        f.write(f"\n===== Scenario={scenario}, Payload={payload_size}, Run={run_id} =====\n{result.stdout}\n")

    if result.returncode != 0:
        print("Hey failed:", result.stderr)
        return None

    return parse_hey_output(result.stdout)


def run_hey_with_rate(scenario=None, payload_size=None, run_id=None, qps=50, concurrency=1, duration="10s", wait=True):
    cmd = [
        "hey", "-z", duration, "-q", str(qps), "-c", str(concurrency), "http://localhost:8000"
    ]

    if wait:
        # Standalone (default)
        result = subprocess.run(cmd, capture_output=True, text=True)
        with open(HEY_DEBUG_LOG, "a") as f:
            f.write(f"\n===== Scenario={scenario}, Payload={payload_size}, Run={run_id} =====\n")
            f.write(result.stdout)
        if result.returncode != 0:
            print("Hey failed:", result.stderr)
            return None
        return parse_hey_output(result.stdout)
    else:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_hey_output(output):
    rps = latency = None
    for line in output.splitlines():
        if line.startswith("  Requests/sec"):
            rps = float(line.split()[1])
        if line.startswith("  Average"):
            try:
                latency = float(line.split()[1]) * 1000
            except ValueError:
                latency = None
    return (rps, latency) if rps and latency else (None, None)


def sample_resources(scenario, payload_size, run_id, stop_event):
    def find_proc(name_match):
        for p in psutil.process_iter(attrs=["pid", "cmdline"]):
            if name_match in " ".join(p.info["cmdline"]):
                return psutil.Process(p.info["pid"])
        return None

    wasm_proc = find_proc("wasmcloud")
    nats_proc = find_proc("nats-server")
    
    if not wasm_proc and not nats_proc:
        raise RuntimeError("Missing required processes: wasmcloud and nats-server")
    elif not wasm_proc:
        raise RuntimeError("Missing required process: wasmcloud")
    elif not nats_proc:
        raise RuntimeError("Missing required process: nats-server")

    # Prime the sampling window
    wasm_proc.cpu_percent(interval=None)
    nats_proc.cpu_percent(interval=None)
    for child in wasm_proc.children(recursive=True):
        child.cpu_percent(interval=None)

    time.sleep(RESOURCE_SAMPLE_INTERVAL)

    start = time.time()
    while not stop_event.is_set():
        # wasmCloud spawns providers as separate processes
        # CPU and memory for wasmcloud + its children
        wasm_cpu = wasm_proc.cpu_percent(interval=None)
        wasm_mem = wasm_proc.memory_info().rss

        for child in wasm_proc.children(recursive=True):
            try:
                wasm_cpu += child.cpu_percent(interval=None)
                wasm_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # CPU and memory for nats server
        nats_cpu = nats_proc.cpu_percent(interval=None)
        nats_mem = nats_proc.memory_info().rss

        total_cpu = wasm_cpu + nats_cpu
        total_mem_mb = (wasm_mem + nats_mem) / 1024 ** 2
        timestamp = round(time.time() - start, 1)

        write_result(
            RESOURCE_PROFILE_CSV,
            ["scenario", "payload_size", "run_id", "timestamp_s", "cpu_percent", "memory_mb"],
            [scenario, payload_size, run_id, timestamp, total_cpu, total_mem_mb]
        )

        time.sleep(RESOURCE_SAMPLE_INTERVAL)


# ================== BENCHMARK MODES ==================
def benchmark_throughput_latency(scenario, wasmcloud_bin):
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, RUNS_PER_SIZE + 1):
            print(f"Running scenario={scenario} size={size} run={run_id}")
            # stop_all(scenario)
            start_wasmcloud(scenario, wasmcloud_bin, bench_path, run_id, limit_usage=True)


            result = run_hey(scenario, size, run_id)
            if result:
                rps, latency = result
                write_result(THROUGHPUT_CSV, ["scenario", "payload_size", "run_id", "requests_per_sec", "avg_latency_ms"], [scenario, size, run_id, rps, latency])
            stop_all(scenario)


def benchmark_resource_usage(scenario, wasmcloud_bin, qps=100, concurrency=1, duration="5s"):
    duration_sec = int(duration.strip("s"))
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, RUNS_PER_SIZE + 1):
            print(f"Resource profiling scenario={scenario} size={size} run={run_id}")
            #stop_all(scenario)
            start_wasmcloud(scenario, wasmcloud_bin, bench_path, run_id, limit_usage=False)

            stop_event = ThreadEvent()

            # Start hey using the refactored function (non-blocking mode)
            hey_proc = run_hey_with_rate(
                scenario=scenario,
                payload_size=size,
                run_id=run_id,
                qps=qps,
                concurrency=concurrency,
                duration=duration,
                wait=False  # Don't block
            )

            # Start resource sampling right after hey is launched
            monitor_thread = Thread(target=sample_resources, args=(scenario, size, run_id, stop_event))
            monitor_thread.start()

            stdout, stderr = hey_proc.communicate()
            stop_event.set()
            monitor_thread.join()

            # Log hey output
            with open(HEY_DEBUG_LOG, "a") as f:
                f.write(f"\n===== Scenario={scenario}, Payload={size}, Run={run_id} =====\n")
                f.write(stdout)

            if hey_proc.returncode == 0:
                rps, latency = parse_hey_output(stdout)
                print("Resource profiling successful")
                write_result(
                    RESOURCE_PROFILE_CSV,
                    ["scenario", "payload_size", "run_id", "requests_per_sec", "avg_latency_ms"],
                    [scenario, size, run_id, rps, latency]
                )
            else:
                print(f"Hey failed: {stderr}")

            stop_all(scenario)


def benchmark_baseline_latency(scenario, wasmcloud_bin, qps=100, duration="5s"):
    duration_sec = int(duration.strip("s"))
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)

        for run_id in range(1, BASELINE_LATENCY_RUNS + 1):
            start_wasmcloud(scenario, wasmcloud_bin, bench_path, run_id, limit_usage=False)
            result = run_hey_with_rate(
                scenario=scenario,
                payload_size=size,
                run_id=f"idle{run_id}",
                qps=qps,
                concurrency=1,
                duration=duration
            )
            if result:
                _, latency = result
                write_result(
                    BASELINE_LATENCY_CSV,
                    ["scenario", "payload_size", "run_id", "avg_latency_ms"],
                    [scenario, size, run_id, latency]
                )

        stop_all(scenario)


# ========================= MAIN =========================
if __name__ == "__main__":
    # Clear existing CSV
    for f in [THROUGHPUT_CSV, BASELINE_LATENCY_CSV, RESOURCE_PROFILE_CSV, HEY_DEBUG_LOG]:
        if os.path.exists(f):
            os.remove(f)

    for scenario, wasmcloud_bin in [("bypass", WASM_CLOUD_BYPASS), ("nats", WASM_CLOUD_NATS), ("composed", WASM_CLOUD_NATS)]:
        benchmark_baseline_latency(scenario, wasmcloud_bin)
        benchmark_throughput_latency(scenario, wasmcloud_bin)
        benchmark_resource_usage(scenario, wasmcloud_bin)

