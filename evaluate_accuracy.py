"""
Accuracy of every Core ML variant, measured against an fp32 PyTorch reference.

Supersedes the first check_accuracy.py, which used ONE random-token input and
compared against the fp16 Core ML model. That was too weak: a single sample
missed an input-dependent NaN, and random tokens are not real text.

Per model:
  reference : fp32 PyTorch eager model (same weights as the converted one)
  inputs    : N_WINDOWS windows of 64 tokens from held-out text (the last 10%
              of tinyshakespeare, never seen in training). The 19M "large"
              model is untrained, so it gets random tokens and only the
              agreement metric is meaningful for it.
  metrics   : agree@1 - fraction of positions whose argmax matches the fp32
              reference's argmax
              ppl     - perplexity on the true next token (real-text models only)
              nan     - any NaN in the output
Each variant is evaluated under CPU_ONLY and ALL (ALL may run on the ANE,
which computes in fp16).

Outputs: accuracy_results.csv, accuracy_plot.png, perplexity_plot.png
"""

import csv
import gc
import os

import numpy as np
import torch
import coremltools as ct
import matplotlib.pyplot as plt

from compression import VARIANTS, variant_path
from transformer import DecoderOnlyTransformer, ExportDecoderOnlyTransformer

SEQ = 64
N_WINDOWS = 32
LABELS = ["fp16"] + list(VARIANTS)
COMPUTE_UNITS = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "ALL": ct.ComputeUnit.ALL}

text = open("tinyshakespeare.txt").read()
heldout_text = text[int(0.9 * len(text)):]  # same split as train.py: never trained on


def windows_from_ids(ids, n, seq):
    """n evenly spaced windows of seq+1 tokens: input = w[:seq], target = w[1:]."""
    starts = np.linspace(0, len(ids) - seq - 2, n).astype(int)
    return np.stack([ids[s:s + seq + 1] for s in starts])


# ---- per-model setup: reference model + evaluation windows ----

def setup_small():
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in heldout_text])
    ref = DecoderOnlyTransformer(len(chars), d_model=128, d_k=32, d_v=32, h=4, d_ff=512, n_layers=4, max_len=SEQ)
    ref.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
    return ref.eval(), windows_from_ids(ids, N_WINDOWS, SEQ), True


def setup_large():
    torch.manual_seed(0)  # rebuilds the exact random-init weights convert_large.py converted
    ref = ExportDecoderOnlyTransformer(65, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, n_layers=6, max_len=SEQ)
    rng = np.random.default_rng(0)
    return ref.eval(), rng.integers(0, 65, (N_WINDOWS, SEQ + 1)), False


def setup_openelm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")  # OpenELM uses the Llama-2 tokenizer
    ids = np.array(tok(heldout_text, add_special_tokens=False)["input_ids"])
    hf = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()

    # Every window MUST start with the BOS token. Without it OpenELM's output is
    # garbage for the whole window, not just position 0: on one held-out window,
    # ppl was 50,380 without BOS vs 100 with it (evidence/e5_openelm_bos.txt).
    # Window = [BOS, t0..t62] -> targets [t0..t63].
    body = windows_from_ids(ids, N_WINDOWS, SEQ - 1)  # (N, SEQ) real tokens
    windows = np.concatenate([np.full((N_WINDOWS, 1), tok.bos_token_id), body], axis=1)

    class Ref(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, use_cache=False).logits

    return Ref(hf).eval(), windows, True


CASES = [
    ("small (808K)", "shakespeare_transformer.mlpackage", setup_small),
    ("large (19M)", "large_transformer.mlpackage", setup_large),
    # stock conversion: fused SDPA op, wrong on the ANE (see convert_openelm.py)
    ("OpenELM-270M", "openelm_270m.mlpackage", setup_openelm),
    # explicit attention: same fp32 reference; only weights-only variants exist for it
    ("OpenELM-270M fixed", "openelm_270m_manualattn.mlpackage", setup_openelm),
]


def perplexity(logits, targets):
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    nll = -np.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    return float(np.exp(nll.mean()))


if __name__ == "__main__":
    results = []

    for name, base_path, setup in CASES:
        print(f"=== {name} ===", flush=True)
        ref, windows, real_text = setup()
        x, y = windows[:, :SEQ], windows[:, 1:]

        with torch.no_grad():
            ref_logits = np.concatenate([ref(torch.tensor(w[None], dtype=torch.long)).numpy() for w in x])
        ref_argmax = ref_logits.argmax(-1)
        ref_ppl = perplexity(ref_logits, y) if real_text else None
        print(f"  fp32 PyTorch reference{'':14s} ppl={ref_ppl if ref_ppl is None else round(ref_ppl, 3)}", flush=True)
        results.append({"model": name, "label": "fp32 (PyTorch)", "compute_unit": "-",
                        "agree_at_1": 1.0, "ppl": ref_ppl, "has_nan": False})
        del ref
        gc.collect()

        for label in LABELS:
            path = variant_path(base_path, label)
            if not os.path.exists(path):
                print(f"  {label:11s} missing - skipped")
                continue
            for cu_label, cu in COMPUTE_UNITS.items():
                ml = ct.models.MLModel(path, compute_units=cu)
                logits = np.concatenate([
                    ml.predict({"tokens": w[None].astype(np.int32)})["logits"] for w in x
                ])
                del ml
                gc.collect()  # loading many models without freeing them segfaulted the first version

                has_nan = bool(np.isnan(logits).any())
                agree = float((logits.argmax(-1) == ref_argmax).mean())
                ppl = perplexity(logits, y) if real_text and not has_nan else None
                print(f"  {label:11s} {cu_label:9s} agree@1={agree:.3f}  "
                      f"ppl={'n/a' if ppl is None else f'{ppl:.3f}'}  nan={has_nan}", flush=True)
                results.append({"model": name, "label": label, "compute_unit": cu_label,
                                "agree_at_1": agree, "ppl": ppl, "has_nan": has_nan})

    with open("accuracy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "label", "compute_unit", "agree_at_1", "ppl", "has_nan"])
        w.writeheader()
        w.writerows(results)
    print("saved accuracy_results.csv")

    # ---- plots ----
    colors = {"CPU_ONLY": "#4C72B0", "ALL": "#DD8452"}
    fig, axes = plt.subplots(1, len(CASES), figsize=(7 * len(CASES), 5), sharey=True)
    for ax, (name, _, _) in zip(axes, CASES):
        rows = [r for r in results if r["model"] == name and r["compute_unit"] != "-"]
        labels = [l for l in LABELS if any(r["label"] == l for r in rows)]
        x = np.arange(len(labels))
        for i, cu in enumerate(COMPUTE_UNITS):
            vals = [next(r["agree_at_1"] for r in rows if r["label"] == l and r["compute_unit"] == cu) for l in labels]
            ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=cu, color=colors[cu])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("top-1 agreement with fp32 PyTorch")
        ax.legend()
    plt.suptitle(f"Accuracy vs fp32 reference ({N_WINDOWS} x {SEQ}-token windows)")
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=150)
    plt.close(fig)
    print("saved accuracy_plot.png")

    ppl_cases = [c for c in CASES if c[0] != "large (19M)"]
    fig, axes = plt.subplots(1, len(ppl_cases), figsize=(7 * len(ppl_cases), 5))
    for ax, (name, _, _) in zip(axes, ppl_cases):
        rows = [r for r in results if r["model"] == name]
        ref_ppl = next(r["ppl"] for r in rows if r["compute_unit"] == "-")
        labels = [l for l in LABELS if any(r["label"] == l for r in rows)]
        x = np.arange(len(labels))
        for i, cu in enumerate(COMPUTE_UNITS):
            vals = []
            for l in labels:
                r = next(r for r in rows if r["label"] == l and r["compute_unit"] == cu)
                vals.append(r["ppl"] if r["ppl"] is not None else np.nan)
            ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=cu, color=colors[cu])
        ax.axhline(ref_ppl, color="black", linestyle="--", linewidth=1, label=f"fp32 ref ({ref_ppl:.2f})")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(name)
        ax.set_ylabel("perplexity on held-out text (log scale, lower is better)")
        ax.legend()
    plt.suptitle("Perplexity by variant (missing bar = NaN output)")
    plt.tight_layout()
    plt.savefig("perplexity_plot.png", dpi=150)
    plt.close(fig)
    print("saved perplexity_plot.png")
