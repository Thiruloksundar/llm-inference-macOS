"""
Evidence E2: per-op device placement from Core ML's own compute plan
(MLComputePlan), under compute_units=ALL, for every model we benchmarked.

"preferred device" is what Core ML's scheduler picks for each op on THIS Mac.
const ops (weights/literals) are excluded - they aren't executed ops.
"""

import gc
import os
import sys
from collections import Counter, defaultdict

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODELS = [
    "shakespeare_transformer.mlpackage",
    "shakespeare_transformer_flash.mlpackage",
    "shakespeare_transformer_sdpa.mlpackage",
    "large_transformer.mlpackage",
    "openelm_270m.mlpackage",
    "openelm_270m_int8.mlpackage",
    "openelm_270m_int4.mlpackage",
    "openelm_270m_palett4.mlpackage",
    "openelm_270m_int4-wo.mlpackage",
    "openelm_270m_manualattn.mlpackage",
    "openelm_270m_manualattn_int8-wo.mlpackage",
    "openelm_270m_manualattn_int4-wo.mlpackage",
    "openelm_270m_manualattn_palett4-wo.mlpackage",
]

SHORT = {"MLCPUComputeDevice": "CPU", "MLGPUComputeDevice": "GPU", "MLNeuralEngineComputeDevice": "ANE"}


def placement(path):
    ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)  # keep alive: compiled path is tied to it
    plan = MLComputePlan.load_from_path(ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.ALL)
    by_device = Counter()
    op_types_by_device = defaultdict(Counter)
    cost_by_device = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        name = op.operator_name.split(".")[-1]
        if name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = SHORT.get(type(usage.preferred_compute_device).__name__, "?") if usage else "none"
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        by_device[dev] += 1
        op_types_by_device[dev][name] += 1
        if cost is not None:
            cost_by_device[dev] += cost.weight
    del plan, ml
    gc.collect()
    return by_device, op_types_by_device, cost_by_device


for path in MODELS:
    by_device, op_types, cost = placement(path)
    total_cost = sum(cost.values()) or 1.0
    cost_share = {d: f"{100 * c / total_cost:.0f}%" for d, c in cost.items()}
    print(f"{path}")
    print(f"  ops per device: {dict(by_device)}   estimated cost share: {cost_share}")
    for dev in sorted(op_types):
        top = ", ".join(f"{t}x{n}" for t, n in op_types[dev].most_common(8))
        print(f"    {dev}: {top}")
    sys.stdout.flush()
