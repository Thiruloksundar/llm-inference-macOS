"""
Evidence E10: error of OpenELM's layer-0 output on the ANE, per token position.

Uses the per-layer model built by e9. If the causal mask is being ignored,
position i wrongly attends to the (63 - i) future tokens, so error should be
largest at position 0 and ~0 at position 63 (whose causal row already sees
every token). Compares the stock model against the explicit-attention fix.
"""

import os
import sys

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, ".."))
sys.path.insert(0, os.getcwd())
from convert_openelm import patch_rope, SEQ_LEN  # noqa: E402

m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()
patch_rope(m)
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text = open("tinyshakespeare.txt").read()
ids = tok(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"]
x = torch.tensor([[tok.bos_token_id] + ids[5000:5000 + SEQ_LEN - 1]])

with torch.no_grad():
    ref = m(x, use_cache=False, output_hidden_states=True).hidden_states[1][0].numpy()

ml = ct.models.MLModel(os.path.join(HERE, "openelm_270m_hidden_states.mlpackage"), compute_units=ct.ComputeUnit.ALL)
ane = ml.predict({"tokens": x.numpy().astype(np.int32)})["h01"][0].astype(np.float32)
err = np.linalg.norm(ane - ref, axis=-1) / np.linalg.norm(ref, axis=-1)

print("stock OpenELM (fused SDPA), layer-0 output on ALL (ANE) vs fp32, relative error per position:")
for p in (0, 1, 2, 4, 8, 16, 32, 48, 56, 60, 62, 63):
    print(f"  pos {p:2d}: {err[p]:.4f}")
print(f"  correlation of error with number of future tokens (63 - pos): "
      f"{np.corrcoef(err, 63 - np.arange(SEQ_LEN))[0, 1]:.3f}")
