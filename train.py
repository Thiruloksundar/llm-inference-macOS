"""
Train the small or medium model on tinyshakespeare, character level.

Loss: cross-entropy on next-character prediction. At every position the model
outputs a score for each of the 65 characters; the loss is -log(probability the
model gave to the character that actually comes next), averaged over positions.

    python train.py small    -> checkpoint_small.pt
    python train.py medium   -> checkpoint_medium.pt
"""

import sys

import torch
import torch.nn.functional as F

from transformer import CONFIGS, SEQ_LEN, Transformer, load_shakespeare

SIZE = sys.argv[1]
BATCH, STEPS, LR = 32, 3000, 3e-4

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
vocab, train_ids, heldout_ids = load_shakespeare()
model = Transformer(len(vocab), **CONFIGS[SIZE]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def get_batch(ids):
    """Random windows of text; the target is the input shifted one character left."""
    starts = torch.randint(len(ids) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([ids[s:s + SEQ_LEN] for s in starts])
    y = torch.stack([ids[s + 1:s + SEQ_LEN + 1] for s in starts])
    return x.to(device), y.to(device)


def loss_on(x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


for step in range(STEPS + 1):
    loss = loss_on(*get_batch(train_ids))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            heldout = torch.stack([loss_on(*get_batch(heldout_ids)) for _ in range(20)]).mean()
        model.train()
        print(f"step {step:4d}  train loss {loss.item():.3f}  held-out loss {heldout.item():.3f}", flush=True)

torch.save(model.state_dict(), f"checkpoint_{SIZE}.pt")
print(f"saved checkpoint_{SIZE}.pt")
