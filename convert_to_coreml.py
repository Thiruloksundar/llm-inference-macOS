"""
Convert the trained DecoderOnlyTransformer (checkpoint.pt) to Core ML.

Steps: load model with matching hyperparams -> trace with a fixed sequence
length -> convert -> save -> verify the Core ML output matches the PyTorch
output before doing anything else with it.

The fixed length is required here by the conversion itself (the dynamic
x.size(1) path traces to an aten::Int node coremltools can't convert - see
ExportDecoderOnlyTransformer). It is not an ANE requirement: coremltools'
docs recommend fixed/enumerated shapes for best performance, and say
range-shaped models can run on the ANE with the "Infrequent" reshape hint
(iOS 17.4+): https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
"""

import numpy as np
import torch
import coremltools as ct

from transformer import ExportDecoderOnlyTransformer

# ---- must match train.py exactly, or load_state_dict will shape-mismatch ----
vocab_size = 65
block_size = 64
d_model = 128
d_k = d_v = 32
h = 4
d_ff = 512
n_layers = 4

# export-only variant: no x.size()-based dynamic shape logic, fixed to
# block_size - see transformer.py for why. Same trained weights load
# directly since both classes have identical state_dict keys.
model = ExportDecoderOnlyTransformer(
    vocab_size, d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff,
    n_layers=n_layers, max_len=block_size,
)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"), strict=True)
model.eval()

# ---- trace with one fixed shape ----
example_input = torch.randint(0, vocab_size, (1, block_size), dtype=torch.long)

traced_model = torch.jit.trace(model, example_input)

# sanity check: trace shouldn't have changed the model's behavior
with torch.no_grad():
    eager_out = model(example_input)
    traced_out = traced_model(example_input)
trace_diff = (eager_out - traced_out).abs().max().item()
print(f"trace vs eager max diff: {trace_diff:.2e}")
assert trace_diff < 1e-5, "tracing changed model behavior - check for data-dependent control flow"

# ---- convert ----
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="tokens", shape=example_input.shape, dtype=np.int32)],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS18,  # int4 quantization requires iOS18+
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save("shakespeare_transformer.mlpackage")
print("saved shakespeare_transformer.mlpackage")

# ---- verify: Core ML output should closely match PyTorch output ----
coreml_input = {"tokens": example_input.numpy().astype(np.int32)}
coreml_out = mlmodel.predict(coreml_input)["logits"]
pytorch_out = eager_out.numpy()

diff = np.abs(coreml_out - pytorch_out)
print(f"CoreML vs PyTorch max diff: {diff.max():.4f}")
print(f"CoreML vs PyTorch mean diff: {diff.mean():.4f}")
print("(some diff is expected - Core ML runs in fp16 by default; "
      "a few tenths, not orders of magnitude off, is normal)")

# quick check for NaNs (the masked_fill(-inf) failure mode mentioned earlier)
if np.isnan(coreml_out).any():
    print("WARNING: NaNs in Core ML output - likely the masked_fill(-inf) issue, "
          "try swapping float('-inf') for -1e9 in causal_mask's use")
else:
    print("no NaNs - looks clean")
