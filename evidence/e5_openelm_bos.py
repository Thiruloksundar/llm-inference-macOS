"""
Evidence E5: OpenELM needs a BOS token at position 0.

Perplexity of the fp32 PyTorch model on held-out tinyshakespeare windows,
with and without a leading <s> (BOS). Also reports the top1-top2 logit gap.

Result: without BOS the model is not uncertain but confidently WRONG
(larger logit gaps, perplexity near vocab size). I expected a flat
distribution here; the data says otherwise. Every OpenELM accuracy number
measured before this script (random tokens, no BOS) is therefore invalid.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SEQ, N = 64, 32
tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
m = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()

text = open("tinyshakespeare.txt").read()
ids = np.array(tok(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"])
starts = np.linspace(0, len(ids) - SEQ - 2, N).astype(int)


def stats(with_bos):
    nlls, margins = [], []
    for s in starts:
        body = ids[s:s + SEQ]                                   # 64 real tokens
        seq = np.concatenate([[tok.bos_token_id], body]) if with_bos else ids[s:s + SEQ + 1]
        x, y = torch.tensor(seq[:SEQ])[None], torch.tensor(seq[1:SEQ + 1])
        with torch.no_grad():
            logits = m(x, use_cache=False).logits[0]
        nlls.append(F.cross_entropy(logits, y, reduction="none"))
        top2 = logits.topk(2, dim=-1).values
        margins.append(top2[:, 0] - top2[:, 1])                 # gap between best and 2nd-best logit
    nll = torch.cat(nlls)
    margin = torch.cat(margins)
    return nll.mean().exp().item(), margin.median().item(), (margin < 0.05).float().mean().item()


for with_bos in (False, True):
    ppl, med_margin, frac_tied = stats(with_bos)
    print(f"{'with BOS   ' if with_bos else 'without BOS'}  ppl={ppl:10.1f}   median top1-top2 logit gap={med_margin:.3f}   "
          f"positions with gap<0.05: {100 * frac_tied:.1f}%   ({N} windows x {SEQ} tokens)")
