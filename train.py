"""
Minimal training loop for DecoderOnlyTransformer on tinyshakespeare,
char-level. Priority here is a real, working checkpoint to hand to
coremltools — not state-of-the-art loss, so this is deliberately small
and fast (should finish in a few minutes on Apple Silicon MPS).
"""

import torch
from transformer import DecoderOnlyTransformer

# ---- data ----
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ---- config (kept small on purpose - correctness + speed over quality) ----
block_size = 64      # context window
batch_size = 32
d_model = 128
d_k = d_v = 32
h = 4
d_ff = 512
n_layers = 4
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(
        vocab_size, d_model=d_model, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff,
        n_layers=n_layers, max_len=block_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        x, y = get_batch("train")
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("saved checkpoint.pt")

    # sample some generated text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=300)
    print("\n--- sample generation ---")
    print(decode(generated[0].tolist()))
