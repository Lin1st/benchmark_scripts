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
import tempfile
import re
from paths import (
    WASMCLOUD_NATS,
    WASMCLOUD_BYPASS,
    WASH,
    THROUGHPUT_CSV,
    BASELINE_LATENCY_CSV,
    RESOURCE_PROFILE_CSV,
    RESOURCE_VS_RPS_CSV,
    BENCH_DIR_TEMPLATE,
    HEY_DEBUG_LOG,
    VEGETA_DEBUG_LOG,
)


# Benchmark settings
PAYLOAD_SIZES = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
PAYLOAD_SIZES = [0, 512]
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


def find_named_process(name_match):
    for p in psutil.process_iter(attrs=["pid", "cmdline"]):
        if name_match in " ".join(p.info["cmdline"]):
            return psutil.Process(p.info["pid"])
    return None


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


def run_hey_with_rate(scenario=None, payload_size=None, run_id=None, qps=50, duration="10s", wait=True):
    #cmd = [
    #    "hey", "-z", duration, "-q", str(qps), "-c", str(concurrency), "http://localhost:8000"
    #]
    cmd = [
        "hey", "-z", duration, "-q", str(qps), "http://localhost:8000"
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


def run_vegeta(scenario=None, payload_size=None, run_id=None, rate=50, duration="10s"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        attack_path = tmp.name

    try:
        attack_cmd = f"echo 'GET http://localhost:8000' | vegeta attack -rate={rate} -duration={duration} -output={attack_path}"
        attack_result = subprocess.run(attack_cmd, shell=True, capture_output=True, text=True)
        if attack_result.returncode != 0:
            print("Vegeta attack failed:", attack_result.stderr)
            return None, None

        report_result = subprocess.run(
            ["vegeta", "report", attack_path],
            capture_output=True,
            text=True
        )
        if report_result.returncode != 0:
            print("Vegeta report failed:", report_result.stderr)
            return None, None

        report_output = report_result.stdout

        with open(VEGETA_DEBUG_LOG, "a") as f:
            f.write(f"\n===== [VEGETA] Scenario={scenario}, Payload={payload_size}, Run={run_id} =====\n")
            f.write(report_output)

        return parse_vegeta_output(report_output)

    finally:
        if os.path.exists(attack_path):
            os.remove(attack_path)


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


def parse_vegeta_output(output):
    rps = None
    latency = None
    time_unit_multipliers = {
        "ns": 1e-6,
        "us": 1e-3,
        "µs": 1e-3,
        "ms": 1,
        "s": 1e3,
        "m": 6e4,
        "h": 3.6e6,
    }

    for line in output.splitlines():
        # Parse RPS (from 'throughput' line)
        if "throughput" in line.lower():
            try:
                rps = float(line.strip().split(",")[-1])
            except Exception as e:
                print("Failed to parse RPS:", e)
                continue

        # Parse mean latency
        elif line.startswith("Latencies"):
            try:
                parts = line.split("]")[-1].strip().split(",")
                mean_str = parts[1].strip()  # 2nd value is mean
                match = re.match(r"([\d.]+)([a-zµ]+)", mean_str)
                if match:
                    val, unit = match.groups()
                    multiplier = time_unit_multipliers.get(unit, None)
                    if multiplier is not None:
                        latency = float(val) * multiplier
            except Exception as e:
                print("Failed to parse latency:", e)
                continue

    return rps, latency


def setup_resource_sampling():
    wasm_proc = find_named_process("wasmcloud")
    nats_proc = find_named_process("nats-server")

    if not wasm_proc or not nats_proc:
        raise RuntimeError("Missing required processes: wasmcloud and/or nats-server")

    wasm_proc.cpu_percent(interval=None)
    nats_proc.cpu_percent(interval=None)
    for child in wasm_proc.children(recursive=True):
        child.cpu_percent(interval=None)

    return wasm_proc, nats_proc


def sample_resource_snapshot(wasm_proc, nats_proc):
    wasm_cpu = wasm_proc.cpu_percent(interval=None)
    wasm_mem = wasm_proc.memory_info().rss

    # wasmCloud spawns providers as separate processes
    for child in wasm_proc.children(recursive=True):
        try:
            wasm_cpu += child.cpu_percent(interval=None)
            wasm_mem += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    nats_cpu = nats_proc.cpu_percent(interval=None)
    nats_mem = nats_proc.memory_info().rss

    total_cpu = wasm_cpu + nats_cpu
    total_mem_mb = (wasm_mem + nats_mem) / 1024 ** 2

    return total_cpu, total_mem_mb


def monitor_and_record_resource_usage(
    wasmcloud_bin,
    bench_path,
    scenario,
    size,
    run_id,
    qps,
    duration,
    output_csv,
    use_vegeta=False
):
    start_wasmcloud(scenario, wasmcloud_bin, bench_path, run_id, limit_usage=False)
    stop_event = ThreadEvent()
    samples = []

    def sampler():
        try:
            wasm_proc, nats_proc = setup_resource_sampling()
            time.sleep(RESOURCE_SAMPLE_INTERVAL)
            while not stop_event.is_set():
                cpu, mem = sample_resource_snapshot(wasm_proc, nats_proc)
                samples.append((cpu, mem))
                time.sleep(RESOURCE_SAMPLE_INTERVAL)
        except RuntimeError as e:
            print(f"Resource sampling error: {e}")

    monitor_thread = Thread(target=sampler)
    monitor_thread.start()

    rps = None
    latency = None
    if use_vegeta:
        rps, latency = run_vegeta(
            scenario=scenario,
            payload_size=size,
            run_id=run_id,
            rate=qps,
            duration=duration
        )
    else:
        hey_proc = run_hey_with_rate(
            scenario=scenario,
            payload_size=size,
            run_id=run_id,
            qps=qps,
            duration=duration,
            wait=False
        )
        stdout, stderr = hey_proc.communicate()
        if hey_proc.returncode == 0:
            rps, _ = parse_hey_output(stdout)
        else:
            print(f"Hey failed: {stderr}")

    stop_event.set()
    monitor_thread.join()

    if rps and samples:
        avg_cpu = sum(s[0] for s in samples) / len(samples)
        avg_mem = sum(s[1] for s in samples) / len(samples)
        write_result(
            output_csv,
            ["scenario", "payload_size", "qps", "run_id", "requests_per_sec", "avg_latency_ms", "cpu_percent", "memory_mb"],
            [scenario, size, qps, run_id, rps, latency, avg_cpu, avg_mem]
        )
    else:
        print("No resource samples collected or RPS/latency unavailable.")

    stop_all(scenario)


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


def benchmark_resource_usage(scenario, wasmcloud_bin, qps=10, duration="5s"):
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, RUNS_PER_SIZE + 1):
            # Auto-adjust concurrency based on QPS
            #concurrency = max(1, min(qps // 10, 100))
            print(f"Resource profiling scenario={scenario} size={size} run={run_id}")
            monitor_and_record_resource_usage(
                wasmcloud_bin=wasmcloud_bin,
                bench_path=bench_path,
                scenario=scenario,
                size=size,
                run_id=run_id,
                qps=qps,
                #concurrency=concurrency,
                duration=duration,
                output_csv=RESOURCE_PROFILE_CSV
            )


def benchmark_baseline_latency(scenario, wasmcloud_bin, qps=10, duration="1s"):
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


def benchmark_resource_vs_rps(scenario, wasmcloud_bin, duration="1s"):
    sizes = [0, 65536]
    qps_values = [10, 50]

    for size in sizes:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for qps in qps_values:
            for run_id in range(1, RUNS_PER_SIZE + 1):
                print(f"Resource vs RPS scenario={scenario} size={size} qps={qps} run={run_id}")
                monitor_and_record_resource_usage(
                    wasmcloud_bin=wasmcloud_bin,
                    bench_path=bench_path,
                    scenario=scenario,
                    size=size,
                    run_id=run_id,
                    qps=qps,
                    duration=duration,
                    output_csv=RESOURCE_VS_RPS_CSV,
                    use_vegeta=True
                )


# ========================= MAIN =========================
if __name__ == "__main__":
    # Clear existing CSV
    for f in [THROUGHPUT_CSV, BASELINE_LATENCY_CSV, RESOURCE_PROFILE_CSV, RESOURCE_VS_RPS_CSV, HEY_DEBUG_LOG, VEGETA_DEBUG_LOG]:
        if os.path.exists(f):
            os.remove(f)

    for scenario, wasmcloud_bin in [("bypass", WASMCLOUD_BYPASS), ("nats", WASMCLOUD_NATS), ("composed", WASMCLOUD_NATS)]:
        #benchmark_baseline_latency(scenario, wasmcloud_bin)
        #benchmark_throughput_latency(scenario, wasmcloud_bin)
        #benchmark_resource_usage(scenario, wasmcloud_bin)
        benchmark_resource_vs_rps(scenario, wasmcloud_bin)

