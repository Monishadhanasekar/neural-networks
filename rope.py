import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
DROPOUT = 0.1

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