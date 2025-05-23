# Benchmarking Scripts for wasmCloud Performance Evaluation

This repository contains scripts and tools to automate benchmarking experiments for evaluating wasmCloud invocation performance under various load conditions. It includes HTTP load generators, data collection logic, system resource tracking, and result visualization.

## Overview

The benchmarking suite includes:

- `run_benchmark.py`: Main automation script that performs deployment, warm-up, load generation, and result recording.
- `path.py`: Path utilities and configuration helpers.
- `analyze_results.py`: Script for parsing CSVs and generating visualizations using pandas and matplotlib.


### Load Generators

Two HTTP benchmarking tools are used and should be installed:

- **[hey](https://github.com/rakyll/hey)**: Concurrency-based load generator.
- **[Vegeta](https://github.com/tsenart/vegeta)**: Rate-based load generator.


### PSS (Proportional Set Size) Memory Sampling

Resource sampling includes memory metrics using psutil.
To collect PSS values, root permissions are required.
Please run benchmarking with sudo if accurate memory measurements are needed.


### Debug Logging

To enable verbose logging of load generator output, pass the `--save-load-generator-output` flag when starting the process

```python run_benchmark.py --save-load-generator-output```

Raw logs from hey and Vegeta are saved alongside the parsed results.

### Error Band Computation

Confidence intervals are configurable and can be calculated using either:

- Standard deviation
- T-distribution
- Z-distribution

## Requirements

Install Python dependencies using:

```pip install -r requirements.txt```
