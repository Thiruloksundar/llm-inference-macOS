"""
Evidence E2b: which ops does Core ML say are NOT supported on the ANE?

For each model, lists op types whose `supported_compute_devices` excludes
the Neural Engine. Explains WHY a model lands on GPU instead of ANE.
"""

import gc
import os
from collections import Counter

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODELS = [
    "shakespeare_transformer.mlpackage",
    "shakespeare_transformer_sdpa.mlpackage",
    "large_transformer.mlpackage",
    "openelm_270m.mlpackage",
    "openelm_270m_int4.mlpackage",
]


def ane_unsupported(path):
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    plan = MLComputePlan.load_from_path(ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.ALL)
    unsupported, total = Counter(), 0
    for op in plan.model_structure.program.functions["main"].block.operations:
        name = op.operator_name.split(".")[-1]
        if name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        total += 1
        supported = {type(d).__name__ for d in usage.supported_compute_devices}
        if "MLNeuralEngineComputeDevice" not in supported:
            unsupported[name] += 1
    del plan, ml
    gc.collect()
    return unsupported, total


for path in MODELS:
    unsupported, total = ane_unsupported(path)
    print(f"{path}: {sum(unsupported.values())}/{total} executed ops NOT supported on ANE -> {dict(unsupported)}")
