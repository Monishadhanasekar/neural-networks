#Section 0: Setup

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
# macbook : "mps"
print(f"Using device: {device}")

#Section 0.1: Download & Prepare Shakespeare

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists("shakespeare.txt"):
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, "shakespeare.txt")

with open("shakespeare.txt", "r") as f:
    text = f.read()

print(f"Total characters: {len(text):,}")
print(f"First 200 characters:\n{text[:200]}")

#character level tokenization

# Build character vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

# Character <-> Integer mappings
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Encode/decode helpers
def encode(s):
    return [char_to_idx[c] for c in s]

def decode(ids):
    return "".join([idx_to_char[i] for i in ids])

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")
print(f"\nExample encoding:")
print(f"  'Hello' -> {encode('Hello')}")
print(f"  {encode('Hello')} -> '{decode(encode('Hello'))}'")

# Train/val split (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training tokens:   {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")

# Batching: grab random chunks of text
def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i:i+context_length] for i in ix])
    y = torch.stack([d[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)

# Quick test
xb, yb = get_batch("train", batch_size=4, context_length=8)
print(f"Input shape:  {xb.shape}  (batch_size x context_length)")
print(f"Target shape: {yb.shape}")
print(f"\nExample (first sequence):")
print(f"  Input:  {decode(xb[0].tolist())!r}")
print(f"  Target: {decode(yb[0].tolist())!r}")
print(f"  (Target is input shifted by 1 character)")

#RMS Normalization

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm:
    - No mean subtraction
    - No bias/shift parameter
    - Just: x / RMS(x) * learnable_scale
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

# We will also use Dropout throughout the model.
# Dropout randomly zeroes some values during training,
# forcing the model to not rely on any single feature.
# This prevents memorization (overfitting).
DROPOUT = 0.2


# --- Demo ---
demo_x = torch.randn(2, 4, 8) #x shape = (batch, sequence_length, embedding_dim)
norm = RMSNorm(8)
demo_out = norm(demo_x)

print("RMSNorm demo:")
print(f"  Input  - mean: {demo_x.mean():.3f}, std: {demo_x.std():.3f}")
print(f"  Output - mean: {demo_out.mean():.3f}, std: {demo_out.std():.3f}")
print(f"  Input range:  [{demo_x.min():.3f}, {demo_x.max():.3f}]")
print(f"  Output range: [{demo_out.min():.3f}, {demo_out.max():.3f}]")
print(f"\n  Parameters: just a scale vector of size {norm.weight.shape}")

#ROPE: Rotary Positional Embeddings

def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    """
    Precompute cosine and sine tables for RoPE.

    Each pair of dimensions gets a different rotation frequency.
    Low dims  -> fast rotation -> short-range patterns
    High dims -> slow rotation -> long-range patterns
    """
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """
    Apply rotary embeddings to a tensor.

    x: [batch, n_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim // 2]

    For each pair of dimensions (2i, 2i+1):
      rotated_2i   = x_2i * cos - x_2i+1 * sin
      rotated_2i+1 = x_2i * sin + x_2i+1 * cos
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, hd//2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]   # even dims
    x2 = x[..., 1::2]  # odd dims

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.stack([out1, out2], dim=-1).flatten(-2)

    # --- Visualize the rotation frequencies ---
demo_cos, demo_sin = precompute_rope_freqs(head_dim=64, max_seq_len=256)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].imshow(demo_cos.T.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
axes[0].set_xlabel("Position in sequence")
axes[0].set_ylabel("Dimension pair")
axes[0].set_title("RoPE Cosine Table")

axes[1].imshow(demo_sin.T.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
axes[1].set_xlabel("Position in sequence")
axes[1].set_ylabel("Dimension pair")
axes[1].set_title("RoPE Sine Table")

plt.tight_layout()
plt.show()

print("Low dimension pairs (top rows)  -> fast oscillation -> short-range patterns")
print("High dimension pairs (bottom rows) -> slow oscillation -> long-range patterns")

#GQA: Grouped Query Attention with RoPE

device = "mps" if torch.backends.mps.is_available() else "cpu"

def repeat_kv(x, n_rep):
    """
    Repeat KV heads to match the number of query heads.
    x: [batch, n_kv_heads, seq_len, head_dim]
    Returns: [batch, n_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.
    n_heads query heads, n_kv_heads key/value heads.
    Groups of (n_heads // n_kv_heads) query heads share one KV pair.
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        b, seq, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V!)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Repeat KV heads to match Q heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # Dropout on attention weights (regularization)
        weights = F.dropout(weights, p=DROPOUT, training=self.training)

        out = weights @ v

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.o_proj(out)

        # --- Demo: see GQA shapes ---
print("GQA Shape Demo:")
print(f"  Config: d_model=256, n_heads=8, n_kv_heads=2")
print(f"  -> head_dim = 256/8 = 32")
print(f"  -> 8 Q heads, 2 KV heads")
print(f"  -> each KV head shared by 4 Q heads\n")

demo_gqa = GroupedQueryAttention(d_model=256, n_heads=8, n_kv_heads=2).to(device)
demo_cos, demo_sin = precompute_rope_freqs(head_dim=32, max_seq_len=128)
demo_cos, demo_sin = demo_cos.to(device), demo_sin.to(device)

demo_in = torch.randn(2, 16, 256, device=device)
demo_out = demo_gqa(demo_in, demo_cos, demo_sin)
print(f"  Input:  {demo_in.shape}")
print(f"  Output: {demo_out.shape}")

q_params = 256 * (8 * 32)
kv_params = 256 * (2 * 32) * 2
print(f"\n  Q projection params:  {q_params:,}  (full 8 heads)")
print(f"  KV projection params: {kv_params:,}  (only 2 heads)")
print(f"  KV cache savings: {8//2}x smaller than standard MHA!")

