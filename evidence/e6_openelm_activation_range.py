"""
Evidence E6: do OpenELM's activations exceed fp16's range inside RMSNorm?

RMSNorm computes mean(x^2). fp16's max finite value is 65504, so any |x| above
sqrt(65504) ~= 256 makes x^2 overflow to inf when computed in fp16 (as the
ANE does). This hooks every RMSNorm input in the fp32 PyTorch model and
records the largest |x|, with and without a BOS token, on the same held-out
windows used by evaluate_accuracy.py.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SEQ, N = 64, 32
FP16_MAX = 65504.0
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()

norms = [(n, mod) for n, mod in m.named_modules() if type(mod).__name__.endswith("RMSNorm")]
print(f"found {len(norms)} RMSNorm modules")

stats = {}
def hook(name):
    def fn(mod, inp, out):
        x = inp[0].detach().abs()
        cur = stats.get(name, (0.0, None))
        pos = int(x.amax(dim=-1)[0].argmax())  # which token position holds the max
        if x.max().item() > cur[0]:
            stats[name] = (x.max().item(), pos)
    return fn

for name, mod in norms:
    mod.register_forward_hook(hook(name))

text = open("tinyshakespeare.txt").read()
ids = np.array(tok(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"])
starts = np.linspace(0, len(ids) - SEQ - 2, N).astype(int)

for with_bos in (False, True):
    stats.clear()
    for s in starts:
        seq = np.concatenate([[tok.bos_token_id], ids[s:s + SEQ - 1]]) if with_bos else ids[s:s + SEQ]
        with torch.no_grad():
            m(torch.tensor(seq)[None], use_cache=False)
    worst_name, (worst, worst_pos) = max(stats.items(), key=lambda kv: kv[1][0])
    n_over = sum(1 for v, _ in stats.values() if v * v > FP16_MAX)
    print(f"{'with BOS   ' if with_bos else 'without BOS'}  max |RMSNorm input| = {worst:9.1f} "
          f"(x^2 = {worst * worst:.3g}) at token position {worst_pos} in {worst_name}; "
          f"{n_over}/{len(stats)} RMSNorms see x^2 > fp16 max ({FP16_MAX:.0f})")
