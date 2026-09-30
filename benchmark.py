"""
For every converted model: size on disk, where Core ML runs it, and latency.

  size       bytes of the .mlpackage on disk
  placement  under ComputeUnit.ALL, how many ops Core ML's scheduler puts on the
             Neural Engine / GPU / CPU (from MLComputePlan). ALL *allows* the
             Neural Engine; it doesn't guarantee it's used.
  latency    median wall time of MLModel.predict() over 50 calls, after 10
             warm-up calls, on CPU_ONLY, CPU_AND_GPU and ALL

    python benchmark.py   ->  results/benchmark.csv, results/latency.png
"""

import csv
import os
import statistics
import time
from collections import Counter

import numpy as np
import coremltools as ct
import matplotlib.pyplot as plt
from coremltools.models.compute_plan import MLComputePlan

MODELS = ["small", "medium", "openelm"]
VARIANTS = ["naive", "flash", "int8", "int4", "palett4"]
VOCAB = {"small": 65, "medium": 65, "openelm": 32000}
COMPUTE_UNITS = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
                 "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                 "ALL": ct.ComputeUnit.ALL}
DEVICE = {"MLCPUComputeDevice": "cpu", "MLGPUComputeDevice": "gpu", "MLNeuralEngineComputeDevice": "ane"}


def size_mb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files) / 1e6


def placement(path):
    """Count of executed ops per device under ALL (constants skipped)."""
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)  # must stay alive while the plan is read
    plan = MLComputePlan.load_from_path(mlmodel.get_compiled_model_path(), compute_units=ct.ComputeUnit.ALL)
    counts = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if op.operator_name.endswith("const") or usage is None:
            continue
        counts[DEVICE[type(usage.preferred_compute_device).__name__]] += 1
    return counts


def latency_ms(path, compute_unit, tokens):
    mlmodel = ct.models.MLModel(path, compute_units=compute_unit)
    for _ in range(10):
        mlmodel.predict({"tokens": tokens})
    times = []
    for _ in range(50):
        start = time.perf_counter()
        mlmodel.predict({"tokens": tokens})
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for name in MODELS:
        tokens = rng.integers(0, VOCAB[name], (1, 64)).astype(np.int32)  # values don't affect latency
        for variant in VARIANTS:
            path = f"models/{name}_{variant}.mlpackage"
            ops = placement(path)
            row = {"model": name, "variant": variant, "size_mb": round(size_mb(path), 2),
                   "ops_ane": ops["ane"], "ops_gpu": ops["gpu"], "ops_cpu": ops["cpu"]}
            for cu_name, cu in COMPUTE_UNITS.items():
                row[f"ms_{cu_name}"] = round(latency_ms(path, cu, tokens), 3)
            rows.append(row)
            print(row, flush=True)

    with open("results/benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 4.5))
    for ax, name in zip(axes, MODELS):
        model_rows = [r for r in rows if r["model"] == name]
        x = np.arange(len(VARIANTS))
        for i, cu_name in enumerate(COMPUTE_UNITS):
            ax.bar(x + (i - 1) * 0.27, [r[f"ms_{cu_name}"] for r in model_rows], 0.27, label=cu_name)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS)
        ax.set_title(name)
        ax.set_ylabel("latency (ms, median)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/latency.png", dpi=130)
    print("saved results/benchmark.csv, results/latency.png")
