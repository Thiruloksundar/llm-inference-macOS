"""
Convert a much larger DecoderOnlyTransformer to Core ML for comparison
against the small (808K param) Shakespeare model - same benchmark sweep,
to see whether the "quantization overhead exceeds benefit" finding from
the small model holds, or reverses, at scale.

Uses RANDOM weights, deliberately - this script is for latency/memory
benchmarking only, which depends on architecture and dtype, not on
whether the weights are trained. No training needed here.
"""

import numpy as np
import torch
import coremltools as ct

from transformer import ExportDecoderOnlyTransformer

vocab_size = 65
block_size = 64        # keep context length the same as the small model - fair comparison
d_model = 512
d_k = d_v = 64
h = 8
d_ff = 2048
n_layers = 6

torch.manual_seed(0)
model = ExportDecoderOnlyTransformer(
    vocab_size, d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff,
    n_layers=n_layers, max_len=block_size,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"large model params: {n_params:,}")

example_input = torch.randint(0, vocab_size, (1, block_size), dtype=torch.long)

traced_model = torch.jit.trace(model, example_input)
with torch.no_grad():
    eager_out = model(example_input)
    traced_out = traced_model(example_input)
trace_diff = (eager_out - traced_out).abs().max().item()
print(f"trace vs eager max diff: {trace_diff:.2e}")
assert trace_diff < 1e-5

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="tokens", shape=example_input.shape, dtype=np.int32)],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save("large_transformer.mlpackage")
print("saved large_transformer.mlpackage")

coreml_out = mlmodel.predict({"tokens": example_input.numpy().astype(np.int32)})["logits"]
diff = np.abs(coreml_out - eager_out.numpy())
print(f"CoreML vs PyTorch max diff: {diff.max():.4f}, mean diff: {diff.mean():.4f}")
print("NaNs:" , np.isnan(coreml_out).any())
