"""
Latency A/B of the three attention implementations in our small trained model
(same weights; only the attention code differs):

  naive : softmax(QK^T / sqrt(d)) V written out  -> matmul/softmax/matmul
  flash : our tiled online-softmax version        -> 107 -> 395 executed ops, ~3.7x (evidence/e2)
  sdpa  : torch F.scaled_dot_product_attention    -> ONE fused op at iOS18 (evidence/e1)

Per-op device placement for these models is in evidence/e2_compute_plan.txt
(all three run entirely on the GPU under ALL).
"""

import gc
import statistics
import time

import numpy as np
import coremltools as ct

from benchmark import COMPUTE_UNITS, N_RUNS, N_WARMUP

MODELS = {
    "naive": "shakespeare_transformer.mlpackage",
    "flash": "shakespeare_transformer_flash.mlpackage",
    "sdpa": "shakespeare_transformer_sdpa.mlpackage",
}

if __name__ == "__main__":
    example_input = np.random.default_rng(0).integers(0, 65, (1, 64)).astype(np.int32)
    print(f"median latency over {N_RUNS} runs after {N_WARMUP} warmup (ms)")
    print(f"{'attention':10s}" + "".join(f"{cu:>14s}" for cu in COMPUTE_UNITS))
    for name, path in MODELS.items():
        row = []
        for cu in COMPUTE_UNITS.values():
            model = ct.models.MLModel(path, compute_units=cu)
            inp = {"tokens": example_input}
            for _ in range(N_WARMUP):
                model.predict(inp)
            times = []
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                model.predict(inp)
                times.append((time.perf_counter() - t0) * 1000)
            row.append(statistics.median(times))
            del model
            gc.collect()
        print(f"{name:10s}" + "".join(f"{v:14.3f}" for v in row))
