"""
Evidence E8: largest |output| of every submodule in fp32 OpenELM, on held-out
text with BOS. Anything above fp16's max (65504) cannot be represented on a
device that computes in fp16 (the ANE).
"""

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SEQ, N, FP16_MAX = 64, 8, 65504.0
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()

peak = {}  # module name -> (max |out|, token position of max)


def hook(name):
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(t) or t.dim() < 3:
            return
        a = t.detach().abs().float()
        v = a.max().item()
        pos = int(a.flatten(0, -3).amax(dim=-1).amax(dim=0).argmax()) if a.dim() >= 3 else -1
        if v > peak.get(name, (0, 0))[0]:
            peak[name] = (v, pos)
    return fn


for name, mod in m.named_modules():
    if name:
        mod.register_forward_hook(hook(name))

text = open("tinyshakespeare.txt").read()
ids = np.array(tok(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"])
for s in np.linspace(0, len(ids) - SEQ - 2, N).astype(int):
    seq = np.concatenate([[tok.bos_token_id], ids[s:s + SEQ - 1]])
    with torch.no_grad():
        m(torch.tensor(seq)[None], use_cache=False)

over = [(n, v, p) for n, (v, p) in peak.items() if v > FP16_MAX]
print(f"{len(over)} module outputs exceed fp16 max ({FP16_MAX:.0f}):")
for n, v, p in sorted(over, key=lambda t: -t[1]):
    print(f"  {n:45s} max|out|={v:12.1f}  at token position {p}")
print("largest 12 module outputs overall:")
for n, (v, p) in sorted(peak.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  {n:45s} max|out|={v:12.1f}  at token position {p}")
