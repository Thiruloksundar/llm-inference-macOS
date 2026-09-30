"""
Convert the three models to Core ML (fp16), each with naive and flash attention:

    models/{small,medium,openelm}_{naive,flash}.mlpackage

    python convert.py                 # all three models
    python convert.py openelm         # just one
"""

import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

from transformer import ATTENTION, CONFIGS, SEQ_LEN, Transformer

os.makedirs("models", exist_ok=True)


def our_model(size, attention):
    model = Transformer(65, attention=attention, **CONFIGS[size])
    model.load_state_dict(torch.load(f"checkpoint_{size}.pt", map_location="cpu"))
    return model.eval()


class LogitsOnly(nn.Module):
    """HuggingFace models return an output object; tracing needs a plain tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens, use_cache=False).logits


def openelm(attention):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()
    code = sys.modules[type(model).__module__]  # OpenELM's own modeling code, downloaded from HuggingFace

    # Patch 1 - needed to convert at all. OpenELM's rotary embedding reads the
    # sequence length from key.shape[2] and slices its sin/cos tables with it;
    # that shape lookup traces into an op coremltools can't convert. Our input
    # is always SEQ_LEN tokens (no KV cache), so use SEQ_LEN directly.
    def rope_forward(self, q, k):
        self._compute_sin_cos_embeddings(SEQ_LEN, key_device=k.device, key_dtype=torch.float32)
        sin, cos = self._cached_sin[..., :SEQ_LEN, :], self._cached_cos[..., :SEQ_LEN, :]
        q_rot = code._apply_rotary_pos_emb(x=q.float(), pos_sin=sin, pos_cos=cos)
        k_rot = code._apply_rotary_pos_emb(x=k.float(), pos_sin=sin, pos_cos=cos)
        return q_rot.type_as(q), k_rot.type_as(k)

    code.OpenELMRotaryEmbedding.forward = rope_forward

    # Patch 2 - swap in our attention. OpenELM calls torch's
    # scaled_dot_product_attention; converted for iOS18 that becomes one fused
    # Core ML op which ignores the causal mask when it runs on the Neural Engine
    # (top-1 agreement with fp32 dropped to 4%). Our naive attention is the
    # same math as separate ops and runs correctly there; flash is the variant
    # being tested. OpenELM's mask is additive (0 / very negative), like ours.
    attention_fn = ATTENTION[attention]
    patched_F = types.ModuleType("F_patched")
    patched_F.__dict__.update(vars(F))
    patched_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, **_: attention_fn(q, k, v, attn_mask)
    code.F = patched_F

    return LogitsOnly(model).eval()


def convert(model, example, path):
    with torch.no_grad():
        reference = model(example)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=example.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS18,   # needed later for int4
    )
    mlmodel.save(path)

    # sanity check: the Core ML model (run on CPU) should match PyTorch
    check = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    out = check.predict({"tokens": example.numpy().astype(np.int32)})["logits"]
    agree = (out.argmax(-1) == reference.numpy().argmax(-1)).mean()
    print(f"saved {path}: max diff vs PyTorch {np.abs(out - reference.numpy()).max():.3f}, "
          f"top-1 agreement {agree:.3f}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or ["small", "medium", "openelm"]
    torch.manual_seed(0)
    for name in names:
        for attention in ("naive", "flash"):
            if name == "openelm":
                model = openelm(attention)
                # OpenELM expects the BOS token (id 1) first; without it its output is garbage
                example = torch.cat([torch.tensor([[1]]), torch.randint(0, 32000, (1, SEQ_LEN - 1))], dim=1)
            else:
                model = our_model(name, attention)
                example = torch.randint(0, 65, (1, SEQ_LEN))
            convert(model, example, f"models/{name}_{attention}.mlpackage")
