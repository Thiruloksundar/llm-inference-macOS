"""
Evidence E9: at which layer does OpenELM's ANE output diverge?

Converts OpenELM with every hidden state (embedding output, after each of the
16 layers, after the final norm) exposed as an extra Core ML output, runs it
under CPU_ONLY and ALL, and compares each against the fp32 PyTorch model -
separately for token position 0 (BOS, where the massive activation lives)
and positions 1..63.

Caveat: extra outputs force intermediates to be materialized, which could
change how Core ML partitions the graph. The final-logits row checks whether
the ANE is still wrong with this graph.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, ".."))
sys.path.insert(0, os.getcwd())
from convert_openelm import patch_rope, SEQ_LEN  # noqa: E402

OUT_PATH = os.path.join(HERE, "openelm_270m_hidden_states.mlpackage")

model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()
patch_rope(model)


class AllHidden(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x, use_cache=False, output_hidden_states=True)
        return tuple(out.hidden_states) + (out.logits,)


wrapped = AllHidden(model).eval()

tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text = open("tinyshakespeare.txt").read()
ids = tok(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"]
x = torch.tensor([[tok.bos_token_id] + ids[5000:5000 + SEQ_LEN - 1]])

with torch.no_grad():
    ref = [t.numpy() for t in wrapped(x)]
names = [f"h{i:02d}" for i in range(len(ref) - 1)] + ["logits"]
print(f"{len(ref)} outputs: h00 = embeddings, h01..h16 = after each layer "
      f"(h16 is after the final norm in OpenELM's implementation), logits")

if not os.path.exists(OUT_PATH):
    traced = torch.jit.trace(wrapped, x, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=x.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name=n) for n in names],
        minimum_deployment_target=ct.target.iOS18,
    )
    ml.save(OUT_PATH)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


results = {}
for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("ALL", ct.ComputeUnit.ALL)]:
    ml = ct.models.MLModel(OUT_PATH, compute_units=cu)
    out = ml.predict({"tokens": x.numpy().astype(np.int32)})
    results[cu_name] = out
    del ml

print(f"\n{'output':7s} | {'CPU rel.err pos0':>16s} {'pos1..':>8s} | {'ALL rel.err pos0':>16s} {'pos1..':>8s} | "
      f"{'ALL max|x| pos0':>15s} {'fp32 max|x| pos0':>16s}")
for i, n in enumerate(names):
    r = ref[i][0]
    row = [n]
    for cu_name in ("CPU_ONLY", "ALL"):
        o = results[cu_name][n][0].astype(np.float32)
        row += [rel_err(o[0], r[0]), rel_err(o[1:], r[1:])]
    all_o = results["ALL"][n][0].astype(np.float32)
    print(f"{row[0]:7s} | {row[1]:16.4f} {row[2]:8.4f} | {row[3]:16.4f} {row[4]:8.4f} | "
          f"{np.abs(all_o[0]).max():15.1f} {np.abs(r[0]).max():16.1f}")

for cu_name in ("CPU_ONLY", "ALL"):
    agree = (results[cu_name]["logits"][0].argmax(-1) == ref[-1][0].argmax(-1)).mean()
    print(f"{cu_name}: final top-1 agreement with fp32 = {agree:.3f}")
