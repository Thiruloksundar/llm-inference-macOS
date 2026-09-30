"""
Evidence E1: what does coremltools turn torch's scaled_dot_product_attention into?

Converts three tiny attention modules (manual softmax, our tiled flash
version, torch SDPA) at two deployment targets (iOS17, iOS18) and prints
the MIL op types in each converted program. If SDPA shows up as a single
fused op, coremltools preserves it; if it shows up as matmul + softmax,
it has been decomposed.
"""

import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

sys.path.insert(0, "..")
from transformer import FlashSelfAttention  # noqa: E402

B, H, T, D = 1, 4, 64, 32


class Manual(nn.Module):
    def forward(self, q, k, v):
        s = q @ k.transpose(-2, -1) / D ** 0.5
        return s.softmax(dim=-1) @ v


class Flash(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FlashSelfAttention(D)

    def forward(self, q, k, v):
        return self.attn(q, k, v)


class SDPA(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


def op_types(mlmodel):
    prog = mlmodel._mil_program
    return Counter(op.op_type for op in prog.functions["main"].operations)


example = [torch.randn(B, H, T, D) for _ in range(3)]
inputs = [ct.TensorType(name=n, shape=(B, H, T, D)) for n in ("q", "k", "v")]

for target_name, target in [("iOS17", ct.target.iOS17), ("iOS18", ct.target.iOS18)]:
    for name, module in [("manual", Manual()), ("flash", Flash()), ("sdpa", SDPA())]:
        traced = torch.jit.trace(module.eval(), example)
        ml = ct.convert(traced, inputs=inputs, minimum_deployment_target=target,
                        convert_to="mlprogram")
        counts = op_types(ml)
        # numerical check against torch
        ref = module(*example).detach().numpy()
        out = list(ml.predict({n: t.numpy() for n, t in zip(("q", "k", "v"), example)}).values())[0]
        print(f"[{target_name}] {name:6s} total_ops={sum(counts.values()):4d} "
              f"max_diff_vs_torch={np.abs(out - ref).max():.4f}  ops={dict(counts)}")
