"""
Convert Apple's OpenELM-270M (real pretrained weights from HuggingFace) to
Core ML with the same fixed-shape tracing approach as our own model.

Patches, applied before tracing:

1. RoPE (always): the stock OpenELMRotaryEmbedding.forward does
   `key_len = key.shape[2]` and slices its cached sin/cos tables with it,
   which traces to an aten::Int node coremltools can't convert (same bug we
   hit in our own model's x.size(1)). Our export always runs one fixed-length
   forward with no KV cache, so key_len == query_len == SEQ_LEN exactly;
   hardcoding them computes the same values without a runtime shape lookup.

2. --manual-attention (THE FIX for wrong output on the Neural Engine):
   OpenELM calls F.scaled_dot_product_attention with a causal mask. At the
   iOS18 target that becomes one fused Core ML op, and on the ANE that op
   ignores the mask (every mask form, including is_causal=True) - attention
   leaks future tokens and the model's output is wrong (4% top-1 agreement,
   perplexity ~2500 vs 48). The same .mlpackage is correct on CPU.
   Evidence: e9 (diverges in layer 0), e10 (error falls with position),
   e11 (minimal single-layer repro on the ANE). The patch computes the same
   math as explicit matmul/add/softmax/matmul, which the ANE runs correctly.

Two diagnostic patches that were tested and did NOT fix the ANE output
(kept so the negative results are reproducible):
   --safe-rmsnorm : overflow-safe RMSNorm (activations reach ~10,870, so
                    x^2 overflows fp16 - real, but not the cause; e6, e7)
   --finite-mask  : finite causal mask instead of fp32-min (-inf in fp16)

Usage:
    python convert_openelm.py                     -> openelm_270m.mlpackage (stock: wrong on ANE)
    python convert_openelm.py --manual-attention  -> openelm_270m_manualattn.mlpackage (fixed)
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForCausalLM

SEQ_LEN = 64  # same context length as our small/large models


class LogitsOnlyWrapper(nn.Module):
    """torch.jit.trace can't handle HF's ModelOutput dataclass return type directly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, use_cache=False).logits


def patch_rope(model):
    rope_cls = type(model.transformer.layers[0].attn.pos_embedding)
    apply_rope = sys.modules[rope_cls.__module__]._apply_rotary_pos_emb

    def forward(self, query, key):
        key_len = query_len = SEQ_LEN
        assert key.shape[-1] == self.model_dim
        q, k = query.float(), key.float()
        self._compute_sin_cos_embeddings(key_len, key_device=k.device, key_dtype=k.dtype)
        q = apply_rope(x=q, pos_sin=self._cached_sin[..., key_len - query_len:key_len, :],
                       pos_cos=self._cached_cos[..., key_len - query_len:key_len, :])
        k = apply_rope(x=k, pos_sin=self._cached_sin[..., :key_len, :],
                       pos_cos=self._cached_cos[..., :key_len, :])
        return q.type_as(query), k.type_as(key)

    rope_cls.forward = forward


MASK_VALUE = -1e4  # finite and fp16-representable; exp(-1e4) is exactly 0 after softmax


def patch_causal_mask_finite(model):
    """
    Stock OpenELM builds its additive causal mask as triu(ones) * finfo(float32).min
    (-3.4e38). Converted to fp16 that becomes -inf. On the ANE, the fused
    scaled_dot_product_attention op then behaves as if there were no mask:
    per-position error in layer 0 falls from 1.05 at position 0 to 0.01 at
    position 63 (evidence/e10), which is the signature of attention leaking
    future tokens. Use a finite value instead. Fixed shape, no padding - same
    assumptions as the RoPE patch.
    """
    model_cls = type(model.transformer)

    def _update_causal_mask(self, attention_mask, input_tensor):
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1) * MASK_VALUE
        return mask[None, None].to(input_tensor.dtype)

    model_cls._update_causal_mask = _update_causal_mask


def patch_manual_attention(model):
    """
    Replace OpenELM's call to F.scaled_dot_product_attention with the explicit
    softmax(QK^T/sqrt(d) + mask) V. At iOS18 the SDPA call converts to ONE
    fused Core ML op (evidence/e1); the explicit form converts to
    matmul/add/softmax/matmul. Same math - isolates the fused op.
    """
    import types
    import torch.nn.functional as F

    openelm_module = sys.modules[type(model).__module__]

    def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, **_):
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        return scores.softmax(dim=-1) @ v

    shim = types.SimpleNamespace(**{n: getattr(F, n) for n in dir(F) if not n.startswith("__")})
    shim.scaled_dot_product_attention = manual_sdpa
    openelm_module.F = shim


def patch_rmsnorm_overflow_safe(model):
    norm_cls = type(model.transformer.norm)

    def _norm(self, x):
        a = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        xs = x / a
        return xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + (self.eps / a) / a)

    norm_cls._norm = _norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-rmsnorm", action="store_true",
                        help="overflow-safe RMSNorm (tested: does NOT fix the ANE output, see evidence/e7)")
    parser.add_argument("--finite-mask", action="store_true",
                        help="finite causal mask instead of -3.4e38 (tested: does NOT fix the ANE output)")
    parser.add_argument("--manual-attention", action="store_true",
                        help="explicit matmul/softmax attention instead of fused SDPA")
    args = parser.parse_args()
    suffix = "".join(["_safenorm" if args.safe_rmsnorm else "",
                      "_finitemask" if args.finite_mask else "",
                      "_manualattn" if args.manual_attention else ""])
    out_path = f"openelm_270m{suffix}.mlpackage"

    model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()
    model.config.use_cache = False
    print(f"OpenELM-270M params: {sum(p.numel() for p in model.parameters()):,}")

    wrapped = LogitsOnlyWrapper(model).eval()
    torch.manual_seed(0)
    # BOS first: OpenELM's output is garbage without it (evidence/e5)
    example_input = torch.cat([torch.tensor([[1]]), torch.randint(0, model.config.vocab_size, (1, SEQ_LEN - 1))], 1)
    with torch.no_grad():
        original_out = wrapped(example_input)

    patch_rope(model)
    if args.safe_rmsnorm:
        patch_rmsnorm_overflow_safe(model)
    if args.finite_mask:
        patch_causal_mask_finite(model)
    if args.manual_attention:
        patch_manual_attention(model)
    with torch.no_grad():
        patched_out = wrapped(example_input)
    patch_diff = (original_out - patched_out).abs().max().item()
    print(f"patched vs original PyTorch max diff: {patch_diff:.2e}")
    assert patch_diff < 1e-3, "patches must not change the model's output"

    traced = torch.jit.trace(wrapped, example_input)
    with torch.no_grad():
        trace_diff = (patched_out - traced(example_input)).abs().max().item()
    print(f"trace vs eager max diff: {trace_diff:.2e}")
    assert trace_diff < 1e-4

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=example_input.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(out_path)
    print(f"saved {out_path}")

    for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("ALL", ct.ComputeUnit.ALL)]:
        ml = ct.models.MLModel(out_path, compute_units=cu)
        out = ml.predict({"tokens": example_input.numpy().astype(np.int32)})["logits"]
        agree = (out.argmax(-1) == original_out.numpy().argmax(-1)).mean()
        print(f"{cu_name:8s} vs PyTorch: max diff {np.abs(out - original_out.numpy()).max():.3f}, "
              f"top-1 agreement {agree:.3f}, NaN {np.isnan(out).any()}")
        del ml
