"""
Convert the small trained transformer with each attention implementation
(flash = our tiled online-softmax, sdpa = torch's scaled_dot_product_attention)
using the SAME trained weights as shakespeare_transformer.mlpackage (naive).
Only the attention implementation differs, so the three can be A/B'd directly.
"""

import numpy as np
import torch
import coremltools as ct

from transformer import ExportDecoderOnlyTransformer

CFG = dict(vocab_size=65, d_model=128, d_k=32, d_v=32, h=4, d_ff=512, n_layers=4, max_len=64)

torch.manual_seed(0)
example_input = torch.randint(0, CFG["vocab_size"], (1, CFG["max_len"]), dtype=torch.long)
naive_ml = ct.models.MLModel("shakespeare_transformer.mlpackage")
naive_out = naive_ml.predict({"tokens": example_input.numpy().astype(np.int32)})["logits"]

for impl in ("flash", "sdpa"):
    model = ExportDecoderOnlyTransformer(attn_impl=impl, **CFG)
    model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"), strict=True)
    model.eval()

    with torch.no_grad():
        eager_out = model(example_input)
    traced = torch.jit.trace(model, example_input)
    with torch.no_grad():
        trace_diff = (eager_out - traced(example_input)).abs().max().item()
    assert trace_diff < 1e-5, f"{impl}: tracing changed behavior ({trace_diff})"

    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=example_input.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )
    path = f"shakespeare_transformer_{impl}.mlpackage"
    ml.save(path)

    out = ml.predict({"tokens": example_input.numpy().astype(np.int32)})["logits"]
    print(f"{impl}: trace diff {trace_diff:.1e} | CoreML vs PyTorch max {np.abs(out - eager_out.numpy()).max():.4f} "
          f"| vs naive CoreML max {np.abs(out - naive_out).max():.4f} | NaN {np.isnan(out).any()} | saved {path}")
