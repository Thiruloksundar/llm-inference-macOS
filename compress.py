"""
Compress each model's naive-attention version three ways:

  int8    linear quantization to 8 bits: w ~= scale * round(w / scale),
          one scale per output channel (row of the weight matrix)
  int4    linear quantization to 4 bits, one scale per block of 32 weights
          (a single scale per row is too coarse with only 16 levels)
  palett4 palettization: k-means picks 16 representative values (a lookup
          table) and every weight is stored as a 4-bit index into it;
          one table per group of 16 output channels

Only constants that feed linear / matmul / gather ops are compressed - the
weight matrices and the embedding table. Compressing everything would also hit
the causal mask (its -inf entries crash k-means), the positional tables and
the norm scales.

    models/{small,medium,openelm}_{int8,int4,palett4}.mlpackage

    python compress.py              # all three models
    python compress.py openelm      # just one (k-means on OpenELM takes ~7 min)
"""

import sys
import time

import coremltools as ct
import coremltools.optimize.coreml as cto

WEIGHT_OPS = {"linear", "matmul", "gather"}

METHODS = {
    "int8": (cto.linear_quantize_weights,
             cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")),
    "int4": (cto.linear_quantize_weights,
             cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                                         granularity="per_block", block_size=32)),
    "palett4": (cto.palettize_weights,
                cto.OpPalettizerConfig(mode="kmeans", nbits=4,
                                       granularity="per_grouped_channel", group_size=16)),
}


def weights_only(mlmodel, op_config):
    """Apply op_config only to constants whose every consumer is a weight op."""
    weights = cto.get_weights_metadata(mlmodel, weight_threshold=512)
    selected = {name: op_config for name, w in weights.items()
                if w.child_ops and {c.op_type for c in w.child_ops} <= WEIGHT_OPS}
    return cto.OptimizationConfig(op_name_configs=selected)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["small", "medium", "openelm"]:
        base = ct.models.MLModel(f"models/{name}_naive.mlpackage")
        for method, (compress, op_config) in METHODS.items():
            start = time.time()
            compress(base, config=weights_only(base, op_config)).save(f"models/{name}_{method}.mlpackage")
            print(f"saved models/{name}_{method}.mlpackage ({time.time() - start:.0f}s)", flush=True)
