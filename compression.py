"""
Every compression variant used in this project, in one place, so the
benchmark (benchmark.py) and the accuracy eval (evaluate_accuracy.py) are
guaranteed to be testing the same configs.

Two families:

1. "global" variants - what we ran first. One config applied to EVERY
   constant > 512 elements, with coremltools' default granularity
   (int: per-channel, palettization: per-tensor). evidence/e3 shows this
   also compresses non-weight constants: positional tables, RoPE sin/cos,
   RMSNorm scales, and OpenELM's causal mask (which contains -inf).

2. "weights-only" (-wo) variants - config applied only to constants feeding
   linear / matmul / gather ops (the learned weight matrices and the
   embedding table), with finer granularity: int4 per-block (block_size=32,
   which is OpLinearQuantizerConfig's default block size) and k-means
   palettization per-grouped-channel (group_size=16, our choice).
   The docs show these granularities in examples (int4 per_block with
   block_size=128 in a GPTQ example; per_grouped_channel with group_size=4
   in a PyTorch example) but do not recommend specific values. Note the
   docs' own palettization example uses a GLOBAL config - the same pattern
   that sweeps in non-weight constants.
   Docs: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html
         https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html
"""

import coremltools.optimize.coreml as cto

WEIGHT_OP_TYPES = ("linear", "matmul", "gather")
THRESHOLD = 512


def _global(op_config, mlmodel):
    return cto.OptimizationConfig(global_config=op_config)


def _weights_only(op_config, mlmodel):
    """
    Per-constant config: compress a constant only if EVERY op it feeds is a
    weight op. Keyed by the constant's own name (op_name_configs), which the
    library checks before its child-op conflict check.

    Why not op_type_configs={"linear": cfg, ...}? Core ML's converter
    de-duplicates identical constants. In the random-init large model the
    zero bias it inserts for our bias=False projections is identical to the
    all-zero LayerNorm betas, so one constant ends up feeding both `linear`
    and `layer_norm` - op_type_configs then raises a config-conflict error.
    """
    meta = cto.get_weights_metadata(mlmodel, weight_threshold=THRESHOLD)
    selected = {
        name: op_config
        for name, m in meta.items()
        if m.child_ops and {c.op_type for c in m.child_ops} <= set(WEIGHT_OP_TYPES)
    }
    return cto.OptimizationConfig(op_name_configs=selected)


def _q(dtype, **kw):
    return cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=dtype, weight_threshold=THRESHOLD, **kw)


# label -> (compress_fn, config_factory(mlmodel), description)
VARIANTS = {
    "int8":      (cto.linear_quantize_weights, lambda m: _global(_q("int8"), m),
                  "int8, per-channel, global (all consts > 512)"),
    "int4":      (cto.linear_quantize_weights, lambda m: _global(_q("int4"), m),
                  "int4, per-channel, global"),
    "palett4":   (cto.palettize_weights,
                  lambda m: _global(cto.OpPalettizerConfig(mode="uniform", nbits=4, weight_threshold=THRESHOLD), m),
                  "4-bit LUT, uniform, per-tensor, global"),
    "int8-wo":   (cto.linear_quantize_weights, lambda m: _weights_only(_q("int8"), m),
                  "int8, per-channel, weights only"),
    "int4-wo":   (cto.linear_quantize_weights,
                  lambda m: _weights_only(_q("int4", granularity="per_block", block_size=32), m),
                  "int4, per-block(32), weights only"),
    # num_kmeans_workers=1: with >1 workers, the second palettize call in the
    # same process failed with "ValueError: Pool not running".
    "palett4-wo": (cto.palettize_weights,
                   lambda m: _weights_only(cto.OpPalettizerConfig(
                       mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16,
                       num_kmeans_workers=1, weight_threshold=THRESHOLD), m),
                   "4-bit LUT, k-means, per-grouped-channel(16), weights only"),
}


def variant_path(base_path, label):
    return base_path if label == "fp16" else base_path.replace(".mlpackage", f"_{label}.mlpackage")


def make_variant(base_mlmodel, label, out_path):
    fn, config_factory, _ = VARIANTS[label]
    fn(base_mlmodel, config=config_factory(base_mlmodel)).save(out_path)
