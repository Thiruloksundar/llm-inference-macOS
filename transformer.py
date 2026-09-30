"""
Decoder-only Transformer (GPT-style), built from the encoder-decoder version
we verified earlier — SelfAttention, MultiHeadAttention, FeedForward,
causal_mask, and sinusoidal_positional_encoding are unchanged and already
tested. What changed: no encoder, no cross-attention, single embedding
table, single stack of blocks (masked self-attention + FF only). This is
the architecture every on-device LLM you'll discuss (Llama, OpenELM, Phi,
Gemma) actually uses.
"""

import math
import torch
import torch.nn as nn


def sinusoidal_positional_encoding(seq_len, d_model, device=None):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Returns: (seq_len, d_model)
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def causal_mask(seq_len, device=None):
    """Lower-triangular mask: position i may attend to positions <= i only."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)


class SelfAttention(nn.Module):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V."""

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        return attn @ V


class FlashSelfAttention(nn.Module):
    """
    Same scaled dot-product attention as SelfAttention (identical math,
    verified below in __main__), computed with block-wise tiling and
    online softmax instead of materializing the full (seq_q, seq_k) score
    matrix at once - the core Flash Attention idea. Works on any leading
    batch dims since every op is on the last two dimensions.

    Honest expectation-setting: at seq_len=64, the "full attention matrix"
    Flash Attention avoids is only 64x64 - tiny, so this isn't expected to
    win on memory or raw latency here. What's actually being tested is
    whether the different OP SEQUENCE (many small tiled matmuls + a
    running max/sum instead of one big matmul + one softmax) changes how
    Core ML schedules it. Measured result: ~3.7x more executed ops (107 -> 395), slower on every
    compute unit, same (GPU) placement - evidence/e1, e2, e12.

    Masking uses -1e9. Note: this is NOT fp16-safe in itself (fp16's max
    magnitude is 65504, so -1e9 still overflows to -inf when Core ML casts
    it) - it's fine because -inf is the value we want there anyway.

    Correctness assumption: the running max `m` starts at -inf, so every
    query row must see at least one unmasked key in the FIRST block, or a
    fully masked first block would contribute exp(0)=1 per masked key.
    With a causal mask this always holds (key 0 is visible to every row),
    and later fully-masked blocks are harmless (their exp underflows to 0
    once m is finite). For arbitrary masks this implementation would need
    an explicit guard.
    """

    def __init__(self, d_k, block_size=16):
        super().__init__()
        self.d_k = d_k
        self.block_size = block_size

    def forward(self, Q, K, V, mask=None):
        seq_k = K.shape[-2]
        scale = self.d_k ** 0.5

        out = torch.zeros(*Q.shape[:-1], V.shape[-1], dtype=Q.dtype, device=Q.device)
        m = torch.full((*Q.shape[:-1], 1), float("-inf"), dtype=Q.dtype, device=Q.device)
        l = torch.zeros((*Q.shape[:-1], 1), dtype=Q.dtype, device=Q.device)

        for start in range(0, seq_k, self.block_size):
            end = min(start + self.block_size, seq_k)
            k_block = K[..., start:end, :]
            v_block = V[..., start:end, :]

            scores = (Q @ k_block.transpose(-2, -1)) / scale  # (..., seq_q, block)
            if mask is not None:
                block_mask = mask[..., start:end]
                scores = scores.masked_fill(block_mask == 0, -1e9)

            block_max = scores.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m, block_max)

            correction = torch.exp(m - new_m)          # rescale everything seen so far
            exp_scores = torch.exp(scores - new_m)      # this block's contribution

            l = l * correction + exp_scores.sum(dim=-1, keepdim=True)
            out = out * correction + exp_scores @ v_block
            m = new_m

        return out / l


class SDPASelfAttention(nn.Module):
    """
    torch.nn.functional.scaled_dot_product_attention. Same math as
    SelfAttention. When converted with minimum_deployment_target >= iOS18,
    coremltools maps this to ONE fused MIL op (scaled_dot_product_attention)
    instead of matmul/softmax/matmul - see evidence/e1_sdpa_lowering.txt.
    """

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        attn_mask = (mask != 0) if mask is not None else None  # bool: True = may attend
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)


ATTENTION_IMPLS = {"naive": SelfAttention, "flash": FlashSelfAttention, "sdpa": SDPASelfAttention}


class MultiHeadAttention(nn.Module):
    """h heads via one big projection reshaped into h heads, then concatenated."""

    def __init__(self, d_model, d_k, d_v, h, attn_impl="naive"):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.W_q = nn.Linear(d_model, h * d_k, bias=False)
        self.W_k = nn.Linear(d_model, h * d_k, bias=False)
        self.W_v = nn.Linear(d_model, h * d_v, bias=False)
        self.W_o = nn.Linear(h * d_v, d_model, bias=False)
        self.attention = ATTENTION_IMPLS[attn_impl](d_k)

    def forward(self, Q, K, V, mask=None):
        batch, seq_q, _ = Q.shape
        seq_k = K.shape[1]

        q = self.W_q(Q).view(batch, seq_q, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch, seq_k, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch, seq_k, self.h, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq_q, self.h * self.d_v)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class DecoderBlock(nn.Module):
    """Masked self-attention + feed-forward. No cross-attention, no encoder."""

    def __init__(self, d_model, d_k, d_v, h, d_ff, attn_impl="naive"):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h, attn_impl=attn_impl)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attention(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        d_k=32,
        d_v=32,
        h=4,
        d_ff=512,
        n_layers=4,
        max_len=256,
        attn_impl="naive",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_enc", sinusoidal_positional_encoding(max_len, d_model))

        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, d_k, d_v, h, d_ff, attn_impl=attn_impl) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """x: (batch, seq_len) integer token ids. Returns logits (batch, seq_len, vocab_size)."""
        n = x.size(1)
        h = self.embed(x) * (self.d_model ** 0.5) + self.pos_enc[:n]
        mask = causal_mask(n, device=x.device)
        for block in self.blocks:
            h = block(h, mask)
        h = self.final_norm(h)
        return self.output_proj(h)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """idx: (batch, seq_len) starting context. Greedy/temperature sampling, autoregressive."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]  # crop to context window
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last position's logits
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


class ExportDecoderOnlyTransformer(nn.Module):
    """
    Export-only variant of DecoderOnlyTransformer for Core ML conversion.

    Identical computation, but forward() has NO `.size()`-based dynamic
    shape logic - everything is fixed to `max_len` (mask and positional
    encoding are precomputed once, never sliced at call time). x must
    always be exactly (batch, max_len).

    Why this exists: `torch.jit.trace` + coremltools choked on the
    original forward's `n = x.size(1)` -> it gets traced as a dynamic
    shape-derivation node (aten::Int) instead of folding into a plain
    constant, which coremltools' converter can't handle. Fixing the shape
    at export time is the fix for that conversion error. (It is not what
    decides ANE placement: this model runs on the GPU even with fixed
    shapes - evidence/e2 - and coremltools' docs say flexible-shape models
    can use the ANE with the "Infrequent" reshape hint.)

    Loads the same trained weights as DecoderOnlyTransformer - both have
    identical state_dict keys (mask here is a plain attribute, not a
    registered buffer, so it doesn't appear in either checkpoint), so
    `load_state_dict(strict=True)` works directly.
    """

    def __init__(self, vocab_size, d_model, d_k, d_v, h, d_ff, n_layers, max_len, attn_impl="naive"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_enc", sinusoidal_positional_encoding(max_len, d_model))

        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, d_k, d_v, h, d_ff, attn_impl=attn_impl) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # plain attribute, not register_buffer -> doesn't appear in state_dict,
        # so it can't cause a load_state_dict key mismatch against the
        # checkpoint saved by the (mask-free) training-time model
        self.mask = causal_mask(max_len)

    def forward(self, x):
        """x: (batch, max_len) integer token ids, exactly. Returns (batch, max_len, vocab_size) logits."""
        h = self.embed(x) * (self.d_model ** 0.5) + self.pos_enc
        for block in self.blocks:
            h = block(h, self.mask)
        h = self.final_norm(h)
        return self.output_proj(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 65  # matches tinyshakespeare's char-level vocab
    model = DecoderOnlyTransformer(vocab_size, d_model=64, d_k=16, d_v=16, h=4, d_ff=256, n_layers=2, max_len=32)

    batch, seq_len = 2, 10
    x = torch.randint(0, vocab_size, (batch, seq_len))
    logits = model(x)
    print("logits shape:", logits.shape)
    assert logits.shape == (batch, seq_len, vocab_size)
    print("shape check passed")

    loss = logits.reshape(-1, vocab_size).sum()
    loss.backward()
    print("backward pass OK")

    gen = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    print("generate() smoke test, output shape:", gen.shape)

    # ---- flash / sdpa attention: verify numerically equivalent to naive ----
    cfg = dict(d_model=64, d_k=16, d_v=16, h=4, d_ff=256, n_layers=2, max_len=32)
    naive_model = DecoderOnlyTransformer(vocab_size, attn_impl="naive", **cfg).eval()
    with torch.no_grad():
        naive_out = naive_model(x)

    for impl in ("flash", "sdpa"):
        other = DecoderOnlyTransformer(vocab_size, attn_impl=impl, **cfg).eval()
        other.load_state_dict(naive_model.state_dict())  # same weights, only attention impl differs
        with torch.no_grad():
            diff = (naive_out - other(x)).abs().max().item()
        print(f"{impl} vs naive attention max diff: {diff:.2e}")
        assert diff < 1e-4, f"{impl} attention should be numerically equivalent to naive attention"
    print("flash and sdpa attention both match naive attention")
