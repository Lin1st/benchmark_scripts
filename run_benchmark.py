# run_benchmark.py
import argparse
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
    HTTP_PROVIDER_REFERENCE,
    URL,
    THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV,
    BASELINE_LATENCY_CSV,
    RESOURCE_VS_PAYLOAD_SIZE_CSV,
    RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV,
    BENCH_DIR_TEMPLATE,
    HEY_DEBUG_LOG,
    VEGETA_DEBUG_LOG,
)


# Benchmark settings
PAYLOAD_SIZES = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
#PAYLOAD_SIZES = [0, 512]
REQUEST_RATES = [10, 50, 100, 200, 500]
NUM_OF_RUNS = 10
REQUESTS_SENDING_DURATION = "3s"
UNDER_LOAD_CONCURRENCY = 5
RESOURCE_SAMPLE_INTERVAL = 0.5

CONFIG = {
    "save_load_generator_output": False
}

RESOURCE_PROFILES = {
    "unlimited": {
        "cpu_quota": None,           # No CPU limit
        "allowed_cpus": None,        # No core pinning (can float across all CPUs)
        "memory_max": None           # No memory limit
    },
    "baseline": {
        "cpu_quota": None,          # No CPU limit
        "allowed_cpus": "0",        # Single physical core
        "memory_max": None          # No memory limit
    },
    "under_load": {
        "cpu_quota": "100%",        # Full CPU access
        "allowed_cpus": "0,1",      # Two physical cores
        "memory_max": "1024M"
    },
    "constrained": {
        "cpu_quota": "50%",         # Half CPU time
        "allowed_cpus": "0,1",      # Both cores, limited slice
        "memory_max": "1024M"
    },
    "stress": {
        "cpu_quota": "50%",         # Constrained with fewer cores
        "allowed_cpus": "0",        # One core
        "memory_max": "512M"
    }
}


# ================== UTILS ==================
def wait_for_port(host, port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not available after {timeout}s")


def build_systemd_cmd(suffix=None, run_id=None, cpu_quota=None, memory_max=None, allowed_cpus=None):
    if run_id is not None:
        unit_name = get_systemd_unit_name(suffix or "bench", run_id)
    else:
        unit_name = f"wasmbench-{suffix or uuid.uuid4().hex[:8]}"

    cmd = ["systemd-run", "--user", "--scope", f"--unit={unit_name}", "-p", "Delegate=yes"]

    if cpu_quota is not None:
        cmd += ["-p", f"CPUQuota={cpu_quota}"]
    if memory_max is not None:
        cmd += ["-p", f"MemoryMax={memory_max}"]
    if allowed_cpus is not None:
        cmd += ["-p", f"AllowedCPUs={allowed_cpus}"]
    return cmd, unit_name


def get_cgroup_path(unit_name):
    result = subprocess.run(
        ["systemctl", "show", "--user", f"--property=ControlGroup", f"--unit={unit_name}"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split("=")[1]


def get_systemd_unit_name(role: str, run_id: int | str):
    return f"wasmbench-{role}-{run_id}"

def build_systemd_cmd(role=None, scenario=None, cpu_quota=None, memory_max=None, allowed_cpus=None):
    unit_name = get_systemd_unit_name(role or "bench", scenario or uuid.uuid4().hex[:8])

    cmd = ["systemd-run", "--user", "--scope", f"--unit={unit_name}", "-p", "Delegate=yes"]

    if cpu_quota is not None:
        cmd += ["-p", f"CPUQuota={cpu_quota}"]
    if memory_max is not None:
        cmd += ["-p", f"MemoryMax={memory_max}"]
    if allowed_cpus is not None:
        cmd += ["-p", f"AllowedCPUs={allowed_cpus}"]

    return cmd, unit_name


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
    

# Warm-up request is needed because wasmCloud components are lazy-loaded
# This ensures the first real request doesn't include cold-start latency
def warm_up(url):
    for attempt in range(10):
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True, text=True
            )
            if r.stdout.strip() == "200":
                print("Warm-up HTTP request successful")
                return  # Exit early on success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
        time.sleep(2)

    raise RuntimeError(f"Warm-up HTTP request failed repeatedly (no 200 OK from {url})")


# Manage without wadm
def shutdown_scenario_env(scenario, run_id=None):
    subprocess.run(["pkill", "-f", "wasmcloud"], stdout=subprocess.DEVNULL)
    time.sleep(10)
    subprocess.run(["docker", "rm", "-f", "/nats-server"], stdout=subprocess.DEVNULL)
    time.sleep(2)


def stop_benchmark_components_and_remove_links(scenario):
    aliases = {"bypass": ["http", "pong"], "nats": ["http", "pong"], "composed": ["composed"]}.get(scenario, [])
    for alias in aliases:
        subprocess.run([WASH, "stop", "component", alias], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([WASH, "link", "del", "http-server", "wasi", "http"], stdout=subprocess.DEVNULL)
    subprocess.run([WASH, "link", "del", "http", "example", "pong"], stdout=subprocess.DEVNULL)
    time.sleep(2)


# Manage without wadm
# Setup host, NATS server, and provider once for the scenario
def setup_scenario_env(scenario, wasmcloud_bin, provider_reference, resource_profile="unlimited", url=URL):
    profile = RESOURCE_PROFILES[resource_profile]
    host, _ = url.split("://")[1].split(":")

    # Build the systemd-run command and unit name
    cmd, unit_name = build_systemd_cmd(
        role="run",
        scenario=scenario,
        cpu_quota=profile["cpu_quota"],
        memory_max=profile["memory_max"],
        allowed_cpus=profile["allowed_cpus"]
    )

    # Start a shell scope where wasmcloud is run
    subprocess.Popen(cmd + ["bash", "-c", f"""
        CGROUP_PARENT=$(systemctl show --property=ControlGroup --value --user {unit_name});
        docker run --cgroup-parent=$CGROUP_PARENT -d --name nats-server -p 4222:4222 -p 8222:8222 nats:latest -js;
        sleep 30;
        WASMCLOUD_ALLOW_FILE_LOAD=true \
        WASMCLOUD_RPC_HOST={host} \
        WASMCLOUD_CTL_HOST={host} \
        {wasmcloud_bin} --max-components 10
    """])

    # Wait for NATS and wasmcloud to be ready
    wait_for_port(host, 4222)
    time.sleep(60)

    subprocess.run([WASH, "start", "provider", provider_reference, "http-server"])
    time.sleep(2)


# Manage without wadm
# Start scenario-specific components and links
def setup_benchmark_components_and_links(scenario, bench_path, url=URL):
    host, port = url.split("://")[1].split(":")
    port = int(port)

    if scenario == "composed":
        subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "composed", "wasi", "http"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/composed.wasm", "composed"])
    else:
        subprocess.run([WASH, "link", "put", "--interface", "incoming-handler", "http-server", "http", "wasi", "http"])
        subprocess.run([WASH, "link", "put", "--interface", "pingpong", "http", "pong", "example", "pong"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/http-hello2/build/http_hello_world_s.wasm", "http"])
        subprocess.run([WASH, "start", "component", f"{bench_path}/wasmCloud_benchmark/pong/build/pong_s.wasm", "pong"])

    wait_for_port(host, port)


def parse_hey_output(output):
    throughput = latency = None
    for line in output.splitlines():
        if line.startswith("  Requests/sec"):
            throughput = float(line.split()[1])
        if line.startswith("  Average"):
            try:
                latency = float(line.split()[1]) * 1000
            except ValueError:
                latency = None
    return (throughput, latency) if throughput and latency else (None, None)


def run_hey(scenario=None, payload_size=None, run_id=None, concurrency=1, rate_limit_per_worker=None, duration=REQUESTS_SENDING_DURATION, wait=True, config=CONFIG):
    cmd = ["hey", "-z", duration]
    cmd += ["-c", str(concurrency)]

    if rate_limit_per_worker is not None:
        cmd += ["-q", str(rate_limit_per_worker)]

    cmd.append(URL)

    if wait:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if config["save_load_generator_output"]:
            with open(HEY_DEBUG_LOG, "a") as f:
                f.write(f"\n===== [HEY] Scenario={scenario}, Payload={payload_size}, Run={run_id} =====\n")
                f.write(result.stdout)

        if result.returncode != 0:
            print("Hey failed:", result.stderr)
            return None
        return parse_hey_output(result.stdout)
    else:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_vegeta_output(output):
    request_rate = latency = None

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
        if "throughput" in line.lower():
            try:
                throughput = float(line.strip().split(",")[-1])
            except Exception as e:
                print("Failed to parse throughput:", e)

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

    return throughput, latency


def run_vegeta(scenario=None, payload_size=None, run_id=None, request_rate=50, duration=REQUESTS_SENDING_DURATION, url=URL, config=CONFIG):
    # Use a temp file to store attack output (binary format)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        attack_path = tmp.name

    try:
        attack_cmd = f"echo 'GET {url}' | vegeta attack -rate={request_rate} -duration={duration} -output={attack_path}"
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

        if config["save_load_generator_output"]:
            with open(VEGETA_DEBUG_LOG, "a") as f:
                f.write(f"\n===== [VEGETA] Scenario={scenario}, Payload={payload_size}, Run={run_id} =====\n")
                f.write(report_output)

        return parse_vegeta_output(report_output)

    finally:
        if os.path.exists(attack_path):
            os.remove(attack_path)


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
        wasm_cpu += child.cpu_percent(interval=None)
        wasm_mem += child.memory_info().rss

    nats_cpu = nats_proc.cpu_percent(interval=None)
    nats_mem = nats_proc.memory_info().rss

    total_cpu = wasm_cpu + nats_cpu
    total_mem_mib = (wasm_mem + nats_mem) / 1024 ** 2

    return total_cpu, total_mem_mib


def monitor_and_record_resource_usage(
    wasmcloud_bin,
    provider_reference,
    bench_path,
    scenario,
    size,
    run_id,
    request_rate,
    output_csv,
    concurrency=1,
    use_vegeta=False,
    resource_profile="unlimited",
    config=CONFIG
):
    stop_event = ThreadEvent()
    samples = []

    def sampler():
        wasm_proc, nats_proc = setup_resource_sampling()
        time.sleep(RESOURCE_SAMPLE_INTERVAL)
        while not stop_event.is_set():
            cpu, mem = sample_resource_snapshot(wasm_proc, nats_proc)
            samples.append((cpu, mem))
            time.sleep(RESOURCE_SAMPLE_INTERVAL)

    monitor_thread = Thread(target=sampler)
    monitor_thread.start()

    throughput = latency = None
    if use_vegeta:
        throughput, latency = run_vegeta(
            scenario=scenario,
            payload_size=size,
            run_id=run_id,
            request_rate=request_rate,
            config=CONFIG
        )
    else:
        hey_proc = run_hey(
            scenario=scenario,
            payload_size=size,
            run_id=run_id,
            rate_limit_per_worker=request_rate,
            concurrency=concurrency,
            wait=False,
            config=CONFIG
        )
        stdout, stderr = hey_proc.communicate()
        if config["save_load_generator_output"]:
            with open(HEY_DEBUG_LOG, "a") as f:
                f.write(f"\n===== [HEY] Scenario={scenario}, Payload={size}, Run={run_id} =====\n")
                f.write(stdout)

        throughput, _ = parse_hey_output(stdout)

    stop_event.set()
    monitor_thread.join()

    if throughput and samples:
        cpu = sum(s[0] for s in samples) / len(samples)
        mem = sum(s[1] for s in samples) / len(samples)
        headers = ["scenario", "payload_size", "request_rate", "run_id", "throughput", "cpu_percent", "memory_mib"]
        row = [scenario, size, request_rate, run_id, throughput, cpu, mem]

        # Only include latency if we have/need it
        if latency is not None:
            headers.insert(5, "avg_latency_ms")
            row.insert(5, latency)

        write_result(output_csv, headers, row)


# ================== BENCHMARK MODES ==================
def benchmark_throughput_latency(scenario, wasmcloud_bin, provider_reference, url=URL, config=CONFIG):
    setup_scenario_env(scenario, wasmcloud_bin, provider_reference, resource_profile="under_load")
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, NUM_OF_RUNS + 1):
            print(f"Running scenario={scenario} size={size} run={run_id}")
            setup_benchmark_components_and_links(scenario, bench_path)
            warm_up(url)

            result = run_hey(scenario, size, run_id, UNDER_LOAD_CONCURRENCY, config=CONFIG)
            if result:
                throughput, latency = result
                write_result(THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV, ["scenario", "payload_size", "run_id", "throughput", "avg_latency_ms"], [scenario, size, run_id, throughput, latency])
            stop_benchmark_components_and_remove_links(scenario)

    shutdown_scenario_env(scenario)


def benchmark_resource_usage(scenario, wasmcloud_bin, provider_reference, request_rate=10, url=URL, config=CONFIG):
    setup_scenario_env(scenario, wasmcloud_bin, provider_reference, resource_profile="under_load")
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, NUM_OF_RUNS + 1):
            print(f"Resource profiling scenario={scenario} size={size} run={run_id}")
            setup_benchmark_components_and_links(scenario, bench_path)
            warm_up(url)

            monitor_and_record_resource_usage(
                wasmcloud_bin=wasmcloud_bin,
                provider_reference=provider_reference,
                bench_path=bench_path,
                scenario=scenario,
                size=size,
                run_id=run_id,
                request_rate=request_rate,
                concurrency=1,
                output_csv=RESOURCE_VS_PAYLOAD_SIZE_CSV,
                resource_profile="under_load",
                config=CONFIG
            )
            stop_benchmark_components_and_remove_links(scenario)

    shutdown_scenario_env(scenario)


def benchmark_baseline_latency(scenario, wasmcloud_bin, provider_reference, request_rate=10, url=URL, config=CONFIG):
    setup_scenario_env(scenario, wasmcloud_bin, provider_reference, resource_profile="baseline")
    for size in PAYLOAD_SIZES:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for run_id in range(1, NUM_OF_RUNS + 1):
            setup_benchmark_components_and_links(scenario, bench_path)
            warm_up(url)

            result = run_hey(
                scenario=scenario,
                payload_size=size,
                run_id=f"idle{run_id}",
                rate_limit_per_worker=request_rate,
                concurrency=1,
                config=CONFIG
            )
            if result:
                _, latency = result
                write_result(
                    BASELINE_LATENCY_CSV,
                    ["scenario", "payload_size", "run_id", "avg_latency_ms"],
                    [scenario, size, run_id, latency]
                )
            stop_benchmark_components_and_remove_links(scenario)

    shutdown_scenario_env(scenario)


def benchmark_resource_vs_request_rate(scenario, wasmcloud_bin, provider_reference, url=URL, config=CONFIG):
    sizes = [0, 1024]
    setup_scenario_env(scenario, wasmcloud_bin, provider_reference, resource_profile="under_load")

    for size in sizes:
        bench_path = BENCH_DIR_TEMPLATE.format(size)
        for request_rate in REQUEST_RATES:
            for run_id in range(1, NUM_OF_RUNS + 1):
                print(f"Resource vs Request Rate scenario={scenario} size={size} request_rate={request_rate} run={run_id}")
                setup_benchmark_components_and_links(scenario, bench_path)
                warm_up(url)

                monitor_and_record_resource_usage(
                    wasmcloud_bin=wasmcloud_bin,
                    provider_reference=provider_reference,
                    bench_path=bench_path,
                    scenario=scenario,
                    size=size,
                    run_id=run_id,
                    request_rate=request_rate,
                    output_csv=RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV,
                    use_vegeta=True,
                    resource_profile="under_load",
                    config=CONFIG
                )
                stop_benchmark_components_and_remove_links(scenario)

    shutdown_scenario_env(scenario)


# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-load-generator-output",
        action="store_true",
        help="Save hey/vegeta output logs to debug files",
    )
    args = parser.parse_args()
    CONFIG["save_load_generator_output"] = args.save_load_generator_output

    # Clear existing CSV
    for f in [THROUGHPUT_AND_LATENCY_VS_PAYLOAD_SIZE_UNDER_LOAD_CSV, BASELINE_LATENCY_CSV, RESOURCE_VS_PAYLOAD_SIZE_CSV, RESOURCE_AND_LATENCY_VS_REQUEST_RATE_CSV, HEY_DEBUG_LOG, VEGETA_DEBUG_LOG]:
        if os.path.exists(f):
            os.remove(f)

    for scenario, wasmcloud_bin in [("nats", WASMCLOUD_NATS), ("composed", WASMCLOUD_NATS), ("bypass", WASMCLOUD_BYPASS)]:
        benchmark_baseline_latency(scenario, wasmcloud_bin, HTTP_PROVIDER_REFERENCE, url=URL, config=CONFIG)
        benchmark_throughput_latency(scenario, wasmcloud_bin, HTTP_PROVIDER_REFERENCE, url=URL, config=CONFIG)
        benchmark_resource_usage(scenario, wasmcloud_bin, HTTP_PROVIDER_REFERENCE, url=URL, config=CONFIG)
        benchmark_resource_vs_request_rate(scenario, wasmcloud_bin, HTTP_PROVIDER_REFERENCE, url=URL, config=CONFIG)


if __name__ == "__main__":
    main()

