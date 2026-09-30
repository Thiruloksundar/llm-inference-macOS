"""
Evidence E11: minimal reproduction - Core ML's fused scaled_dot_product_attention
op (iOS18 target) on the Neural Engine vs CPU, for different mask forms.

A single attention layer (qkv projection -> attention -> output projection)
with OpenELM-like shapes, fp16 inputs/outputs, random weights. First attempt
used a bare SDPA op with fp32 I/O: Core ML scheduled it on the CPU even under
CPU_AND_NE, so it never tested the ANE. The projections + fp16 I/O give the
scheduler a reason to use the ANE; MLComputePlan confirms where the attention
op actually ran ("runs on" column).
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

B, H, T, D = 1, 12, 64, 64
C = H * D
torch.manual_seed(0)
x = torch.randn(B, T, C)
causal_bool = torch.ones(T, T, dtype=torch.bool).tril()

MASKS = {
    "float, fp32-min (OpenELM's)": torch.triu(torch.ones(T, T), 1)[None, None] * torch.finfo(torch.float32).min,
    "float, finite -1e4": torch.triu(torch.ones(T, T), 1)[None, None] * -1e4,
    "bool": causal_bool[None, None],
    "is_causal=True": "is_causal",
    "no mask (control)": None,
}


class Layer(nn.Module):
    def __init__(self, mask, fused):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.out = nn.Linear(C, C, bias=False)
        self.mask, self.fused = mask, fused

    def forward(self, x):
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        m = self.mask
        if self.fused:
            if isinstance(m, str):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        else:
            s = q @ k.transpose(-2, -1) / D ** 0.5
            if isinstance(m, str):
                m = causal_bool
            if m is not None:
                s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m
            a = s.softmax(-1) @ v
        return self.out(a.transpose(1, 2).reshape(B, T, C))


def attn_device(ml):
    plan = MLComputePlan.load_from_path(ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    devs = set()
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name.split(".")[-1] in ("scaled_dot_product_attention", "softmax"):
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            devs.add(type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", ""))
    return ",".join(sorted(devs)) or "?"


print(f"{'mask form':28s} {'attention':9s} {'runs on':12s} | {'CPU max err':>11s} | {'ANE max err':>11s} "
      f"{'ANE err pos0':>12s} {'ANE err pos63':>13s}")
for mask_name, mask in MASKS.items():
    for fused in (True, False):
        torch.manual_seed(1)  # identical weights for fused and explicit
        mod = Layer(mask, fused).eval()
        try:
            with torch.no_grad():
                ref = mod(x).numpy()
            traced = torch.jit.trace(mod, x)
            ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
                            outputs=[ct.TensorType(name="y", dtype=np.float16)],
                            minimum_deployment_target=ct.target.iOS18)
            path = "/tmp/e11_attn.mlpackage"
            ml.save(path)
            feed = {"x": x.numpy().astype(np.float16)}
            cpu = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            ane = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            out_cpu = cpu.predict(feed)["y"].astype(np.float32)
            out_ane = ane.predict(feed)["y"].astype(np.float32)
            dev = attn_device(ane)
            scale = np.abs(ref).max()
            per_pos = np.abs(out_ane - ref).max(axis=(0, 2)) / scale
            print(f"{mask_name:28s} {'fused' if fused else 'explicit':9s} {dev:12s} | "
                  f"{np.abs(out_cpu - ref).max() / scale:11.4f} | {np.abs(out_ane - ref).max() / scale:11.4f} "
                  f"{per_pos[0]:12.4f} {per_pos[-1]:13.4f}")
            del cpu, ane, ml
        except Exception as e:
            print(f"{mask_name:28s} {'fused' if fused else 'explicit':9s} ERROR: {type(e).__name__}: {str(e)[:160]}")
        gc.collect()

print("(errors are max |out - fp32 torch| / max |fp32 torch|)")
