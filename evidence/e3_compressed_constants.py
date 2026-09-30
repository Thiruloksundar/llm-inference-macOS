"""
Evidence E3: which constants did our compression config (weight_threshold=512,
global config, no op filtering) actually hand to the quantizer/palettizer?

Lists every constant > 512 elements in each fp16 base model, the op types it
feeds, and whether it contains non-finite values. Anything that isn't a real
learned weight (masks, positional tables, RoPE sin/cos) gets compressed too
under a global config - and a -inf entry breaks k-means / uniform LUTs.
"""

import gc
import os

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODELS = ["shakespeare_transformer.mlpackage", "openelm_270m.mlpackage"]
WEIGHT_OPS = {"linear", "matmul", "conv", "gather"}  # ops whose const input is a learned weight

for path in MODELS:
    ml = ct.models.MLModel(path)
    meta = cto.get_weights_metadata(ml, weight_threshold=512)
    print(f"=== {path}: {len(meta)} constants > 512 elements ===")
    n_weight, n_other = 0, 0
    for name, m in meta.items():
        val = np.asarray(m.val)
        child_types = sorted({c.op_type for c in m.child_ops})
        is_weight = bool(set(child_types) & WEIGHT_OPS)
        n_weight += is_weight
        n_other += not is_weight
        nonfinite = int((~np.isfinite(val)).sum())
        if not is_weight or nonfinite:
            print(f"  NON-WEIGHT {name:45s} shape={tuple(val.shape)} feeds={child_types} "
                  f"min={np.nanmin(val):.3g} max={np.nanmax(val):.3g} unique={m.unique_values} non_finite={nonfinite}")
    print(f"  -> {n_weight} learned-weight constants, {n_other} non-weight constants also eligible for compression")
    del ml, meta
    gc.collect()
