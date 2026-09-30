"""
Latency + on-disk size sweep:
    model (small 808K trained / large 19M random-init / OpenELM-270M pretrained)
  x variant (fp16 baseline + every compression variant in compression.py)
  x compute unit (CPU_ONLY / CPU_AND_GPU / ALL)

"ALL" lets Core ML use CPU, GPU or ANE - it does NOT guarantee the ANE is
used. evidence/e2_compute_plan.txt shows our own models run entirely on GPU
under ALL, while OpenELM runs almost entirely on the ANE.

Latency = wall time of MLModel.predict() from Python (includes Python and
input-marshalling overhead), after warmup. Size = .mlpackage bytes on disk,
not runtime memory. Accuracy of each variant is measured separately in
evaluate_accuracy.py - latency alone says nothing about whether a variant works.

Outputs: results.csv, benchmark_plot.png (latency), memory_plot.png (size).
"""

import os
import time
import statistics
import csv

import numpy as np
import coremltools as ct
import matplotlib.pyplot as plt

from compression import VARIANTS, variant_path, make_variant

BLOCK_SIZE = 64
N_WARMUP = 10
N_RUNS = 50

CASES = [
    {"name": "small (808K)", "base_path": "shakespeare_transformer.mlpackage", "vocab_size": 65},
    {"name": "large (19M)", "base_path": "large_transformer.mlpackage", "vocab_size": 65},
    # stock conversion: fused SDPA ignores the causal mask on the ANE -> fast but WRONG on ALL
    {"name": "OpenELM-270M", "base_path": "openelm_270m.mlpackage", "vocab_size": 32000},
    # explicit attention (convert_openelm.py --manual-attention): correct on the ANE.
    # Only the weights-only variants - the global ones are shown broken in evidence/e3, e4.
    {"name": "OpenELM-270M fixed", "base_path": "openelm_270m_manualattn.mlpackage", "vocab_size": 32000,
     "variants": ["int8-wo", "int4-wo", "palett4-wo"]},
]

COMPUTE_UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL": ct.ComputeUnit.ALL,
}

LABELS = ["fp16"] + list(VARIANTS)
COLORS = dict(zip(LABELS, plt.get_cmap("tab10").colors))


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def benchmark_model(path, compute_unit, example_input):
    model = ct.models.MLModel(path, compute_units=compute_unit)
    inp = {"tokens": example_input}
    for _ in range(N_WARMUP):
        model.predict(inp)
    times_ms = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(inp)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "std_ms": statistics.stdev(times_ms),
    }


def plot_facets(results, value_fn, err_fn, ylabel, title, out, by_compute_unit):
    fig, axes = plt.subplots(1, len(CASES), figsize=(8 * len(CASES), 5))
    for ax, case in zip(axes, CASES):
        rows = [r for r in results if r["model"] == case["name"]]
        labels = [l for l in LABELS if any(r["label"] == l for r in rows)]
        if by_compute_unit:
            groups = list(COMPUTE_UNITS)
            x = np.arange(len(groups))
            width = 0.8 / len(labels)
            for i, label in enumerate(labels):
                vals, errs = [], []
                for g in groups:
                    r = next((r for r in rows if r["label"] == label and r["compute_unit"] == g), None)
                    vals.append(value_fn(r) if r and value_fn(r) is not None else 0)
                    errs.append(err_fn(r) if r and err_fn(r) is not None else 0)
                ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width, yerr=errs, capsize=2,
                       label=label, color=COLORS[label])
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend(title="variant", fontsize=8)
        else:
            vals = [value_fn(next(r for r in rows if r["label"] == l)) for l in labels]
            ax.bar(labels, vals, color=[COLORS[l] for l in labels])
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(case["name"])
        ax.set_ylabel(ylabel)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    results = []

    for case in CASES:
        name, base_path = case["name"], case["base_path"]
        print(f"=== {name} ===")
        example_input = rng.integers(0, case["vocab_size"], (1, BLOCK_SIZE)).astype(np.int32)
        base_mlmodel = ct.models.MLModel(base_path)

        paths = {"fp16": base_path}
        for label in case.get("variants", VARIANTS):
            out_path = variant_path(base_path, label)
            if not os.path.exists(out_path):
                print(f"  creating {label}...")
                try:
                    make_variant(base_mlmodel, label, out_path)
                except Exception as e:
                    # a failure here is a finding (e.g. k-means on a -inf constant), not something to hide
                    print(f"  SKIPPING {label}: {type(e).__name__}: {e}")
                    continue
            paths[label] = out_path

        for label, path in paths.items():
            size = dir_size_mb(path)
            for cu_label, cu in COMPUTE_UNITS.items():
                try:
                    stats = benchmark_model(path, cu, example_input)
                    print(f"  {label:11s} {cu_label:12s} median {stats['median_ms']:7.3f} ms "
                          f"(mean {stats['mean_ms']:.3f} +/- {stats['std_ms']:.3f})  {size:7.2f} MB")
                except Exception as e:
                    print(f"  {label:11s} {cu_label:12s} FAILED: {e}")
                    stats = {"mean_ms": None, "median_ms": None, "std_ms": None}
                results.append({"model": name, "label": label, "compute_unit": cu_label,
                                "file_size_mb": size, **stats})

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "label", "compute_unit",
                                               "median_ms", "mean_ms", "std_ms", "file_size_mb"])
        writer.writeheader()
        writer.writerows(results)
    print("saved results.csv")

    plot_facets(results, lambda r: r["median_ms"], lambda r: r["std_ms"], "latency (ms, median; bars = std)",
                "Inference latency by variant and compute unit", "benchmark_plot.png", by_compute_unit=True)
    plot_facets(results, lambda r: r["file_size_mb"], lambda r: None, "size on disk (MB)",
                "Model size on disk by variant", "memory_plot.png", by_compute_unit=False)
