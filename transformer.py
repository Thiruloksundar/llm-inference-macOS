"""
Decoder-only transformer (GPT-style) for the small and medium models, plus the
two attention implementations compared in this project.

The model always takes exactly SEQ_LEN tokens. A fixed length keeps the Core ML
conversion simple: shape-dependent code such as x.size(1) traces into an op
coremltools can't convert.
"""

import math

import torch
import torch.nn as nn

SEQ_LEN = 64
CONFIGS = {
    "small":  dict(d_model=128, n_heads=4, d_ff=512,  n_layers=4),   # ~0.8M params
    "medium": dict(d_model=512, n_heads=8, d_ff=2048, n_layers=6),   # ~19M params
}


# ---------------------------------------------------------------- attention
# Both take q, k, v of shape (batch, heads, T, d_head) and an additive mask:
# 0 where a query may attend to a key, -inf where it may not.

# Both scale q by 1/sqrt(d) BEFORE multiplying by k^T. Same math as scaling the
# product afterwards, but in fp16 (Core ML's default precision) the unscaled
# product can overflow: in the trained medium model it reached 189,730, above
# fp16's max of 65,504, and the Core ML output became NaN.

def naive_attention(q, k, v, mask):
    """softmax(QK^T / sqrt(d) + mask) V, computing the full T x T score matrix."""
    scores = (q / math.sqrt(q.shape[-1])) @ k.transpose(-2, -1)
    return (scores + mask).softmax(dim=-1) @ v


def flash_attention(q, k, v, mask, block_size=16):
    """
    Same result as naive_attention, computed the Flash Attention way: loop over
    the keys in blocks and keep, for every query row, a running max (m), a
    running sum of exponentials (l) and a running output (out). When a new block
    contains a larger score, everything accumulated so far is rescaled by
    exp(old_max - new_max). The full T x T score matrix is never built.

    Assumes every query can see key 0 (true for a causal mask), so the first
    block always gives a finite max.
    """
    q = q / math.sqrt(q.shape[-1])
    opts = dict(dtype=q.dtype, device=q.device)
    m = torch.full(q.shape[:-1] + (1,), float("-inf"), **opts)
    l = torch.zeros(q.shape[:-1] + (1,), **opts)
    out = torch.zeros(q.shape[:-1] + (v.shape[-1],), **opts)

    for start in range(0, k.shape[-2], block_size):
        end = start + block_size
        s = q @ k[..., start:end, :].transpose(-2, -1) + mask[..., start:end]
        new_m = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - new_m)                 # this block, relative to the new max
        correction = torch.exp(m - new_m)        # rescales everything seen so far
        l = l * correction + p.sum(dim=-1, keepdim=True)
        out = out * correction + p @ v[..., start:end, :]
        m = new_m
    return out / l


ATTENTION = {"naive": naive_attention, "flash": flash_attention}


# -------------------------------------------------------------------- model

def sinusoidal_positions(seq_len, d_model):
    """Positional encoding from "Attention Is All You Need"."""
    pos = torch.arange(seq_len).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def causal_mask(seq_len):
    """0 on and below the diagonal, -inf above: token i can only see tokens <= i."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # Q, K, V for all heads in one matmul
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attention = attention

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4).unbind(0)
        y = self.attention(q, k, v, mask)                          # (B, heads, T, d_head)
        return self.out(y.transpose(1, 2).reshape(B, T, C))        # concatenate heads


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, attention)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, mask))   # residual + norm after each sublayer, as in the paper
        return self.norm2(x + self.ff(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, attention="naive"):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, ATTENTION[attention]) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        # fixed tables, not learned; persistent=False keeps them out of the checkpoint
        self.register_buffer("pos", sinusoidal_positions(SEQ_LEN, d_model), persistent=False)
        self.register_buffer("mask", causal_mask(SEQ_LEN), persistent=False)

    def forward(self, tokens):
        """tokens: (batch, SEQ_LEN) character ids -> logits (batch, SEQ_LEN, vocab_size)"""
        x = self.embed(tokens) * self.scale + self.pos
        for block in self.blocks:
            x = block(x, self.mask)
        return self.head(self.norm(x))


# --------------------------------------------------------------------- data

def load_shakespeare():
    """Character-level tinyshakespeare. Returns (vocab, train_ids, heldout_ids);
    the last 10% is held out for evaluation and never trained on."""
    text = open("tinyshakespeare.txt").read()
    vocab = sorted(set(text))
    ids = torch.tensor([vocab.index(c) for c in text])
    split = int(0.9 * len(ids))
    return vocab, ids[:split], ids[split:]


if __name__ == "__main__":
    # flash attention must give the same output as naive attention (same weights)
    torch.manual_seed(0)
    naive = Transformer(65, attention="naive", **CONFIGS["small"]).eval()
    flash = Transformer(65, attention="flash", **CONFIGS["small"]).eval()
    flash.load_state_dict(naive.state_dict())
    x = torch.randint(0, 65, (2, SEQ_LEN))
    with torch.no_grad():
        diff = (naive(x) - flash(x)).abs().max().item()
    print(f"flash vs naive max difference: {diff:.1e}")
    for name, cfg in CONFIGS.items():
        n = sum(p.numel() for p in Transformer(65, **cfg).parameters())
        print(f"{name}: {n:,} parameters")
