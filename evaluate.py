"""
Accuracy of every converted model, compared with the original fp32 PyTorch model.

Inputs: 32 windows of 64 tokens from the held-out last 10% of tinyshakespeare
(never used in training). Core ML models run on the compute unit given on the
command line (default ALL).

Two metrics:
  perplexity      exp(average cross-entropy on the true next token) - the same
                  quantity as the training loss, exponentiated. Roughly "how many
                  tokens the model is effectively choosing between". Measures how
                  good the model still is at predicting real text. Lower is better.
  top-1 agreement fraction of positions where the model's most likely next token
                  equals the fp32 PyTorch model's most likely next token.
                  Measures how closely the converted model still behaves like the
                  original. 1.0 = identical choices.

    python evaluate.py            ->  results/accuracy_ALL.csv, results/accuracy_ALL.png
    python evaluate.py CPU_ONLY   ->  results/accuracy_CPU_ONLY.csv, ...
"""

import csv
import gc
import os
import sys

import numpy as np
import torch
import coremltools as ct
import matplotlib.pyplot as plt

from convert import our_model
from transformer import SEQ_LEN, load_shakespeare

MODELS = ["small", "medium", "openelm"]
VARIANTS = ["naive", "flash", "int8", "int4", "palett4"]
N_WINDOWS = 32
COMPUTE_UNIT = sys.argv[1] if len(sys.argv) > 1 else "ALL"


def windows(ids, n=N_WINDOWS, length=SEQ_LEN + 1):
    starts = np.linspace(0, len(ids) - length, n).astype(int)
    return np.stack([ids[s:s + length] for s in starts])


def reference_and_data(name):
    """fp32 PyTorch model + (inputs, next-token targets) for one model."""
    _, _, heldout = load_shakespeare()
    if name in ("small", "medium"):
        w = windows(heldout.numpy())
        return our_model(name, "naive"), w[:, :-1], w[:, 1:]

    # OpenELM: the original, unpatched model from HuggingFace, and its tokenizer
    # (a public copy of the Llama-2 tokenizer it was trained with).
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    text = open("tinyshakespeare.txt").read()
    ids = np.array(tokenizer(text[int(0.9 * len(text)):], add_special_tokens=False)["input_ids"])
    body = windows(ids, length=SEQ_LEN)
    # OpenELM needs the BOS token first: without it its predictions are garbage
    # (perplexity ~28,600 instead of ~48 in our test).
    w = np.concatenate([np.full((N_WINDOWS, 1), tokenizer.bos_token_id), body], axis=1)
    hf = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M", trust_remote_code=True).eval()
    return (lambda x: hf(x, use_cache=False).logits), w[:, :-1], w[:, 1:]


def perplexity(logits, targets):
    logits = logits.astype(np.float64)
    log_probs = logits - logits.max(-1, keepdims=True)
    log_probs -= np.log(np.exp(log_probs).sum(-1, keepdims=True))
    nll = -np.take_along_axis(log_probs, targets[..., None], axis=-1)
    return float(np.exp(nll.mean()))


def plot(rows, compute_unit):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 4.5))
    for ax, name in zip(axes, MODELS):
        model_rows = [r for r in rows if r["model"] == name and r["variant"] != "fp32 PyTorch"]
        ax.bar([r["variant"] for r in model_rows], [float(r["top1_agreement"]) for r in model_rows])
        ax.set_ylim(0, 1.05)
        ax.set_title(name)
        ax.set_ylabel("top-1 agreement with fp32")
    fig.suptitle(f"compute unit: {compute_unit}")
    fig.tight_layout()
    fig.savefig(f"results/accuracy_{compute_unit}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    for name in MODELS:
        reference, inputs, targets = reference_and_data(name)
        with torch.no_grad():
            ref_logits = np.concatenate([reference(torch.tensor(x[None])).numpy() for x in inputs])
        rows.append({"model": name, "variant": "fp32 PyTorch", "perplexity": round(perplexity(ref_logits, targets), 3),
                     "top1_agreement": 1.0})
        print(rows[-1], flush=True)
        del reference
        gc.collect()

        for variant in VARIANTS:
            mlmodel = ct.models.MLModel(f"models/{name}_{variant}.mlpackage", compute_units=getattr(ct.ComputeUnit, COMPUTE_UNIT))
            logits = np.concatenate([mlmodel.predict({"tokens": x[None].astype(np.int32)})["logits"] for x in inputs])
            del mlmodel
            gc.collect()  # loading many large models without freeing them crashes the process
            rows.append({"model": name, "variant": variant,
                         "perplexity": round(perplexity(logits, targets), 3),
                         "top1_agreement": round(float((logits.argmax(-1) == ref_logits.argmax(-1)).mean()), 3)})
            print(rows[-1], flush=True)

    with open(f"results/accuracy_{COMPUTE_UNIT}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot(rows, COMPUTE_UNIT)
    print(f"saved results/accuracy_{COMPUTE_UNIT}.csv, results/accuracy_{COMPUTE_UNIT}.png")
