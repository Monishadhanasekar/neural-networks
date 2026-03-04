import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

DROPOUT = 0.1

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Two paths:
      gate: SiLU(x @ W_gate) - controls flow
      up:   x @ W_up         - carries information

    Combined: gate * up -> W_down

    SiLU(x) = x * sigmoid(x), a smooth version of ReLU.
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return F.dropout(self.w_down(gate * up), p=DROPOUT, training=self.training)

# --- Demo: Compare activation functions ---
fig, axes = plt.subplots(1, 3, figsize=(14, 3))

x_range = torch.linspace(-5, 5, 200)

axes[0].plot(x_range, F.relu(x_range), linewidth=2)
axes[0].set_title("ReLU (Classic)")
axes[0].set_xlabel("x")
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

axes[1].plot(x_range, F.silu(x_range), linewidth=2, color="orange")
axes[1].set_title("SiLU / Swish (Used in SwiGLU)")
axes[1].set_xlabel("x")
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

gate_demo = F.silu(x_range)
up_demo = x_range * 0.5 + 1
axes[2].plot(x_range, gate_demo * up_demo, linewidth=2, color="green", label="gate x up")
axes[2].plot(x_range, gate_demo, linewidth=1, color="orange", alpha=0.5, linestyle="--", label="gate (SiLU)")
axes[2].plot(x_range, up_demo, linewidth=1, color="blue", alpha=0.5, linestyle="--", label="up (signal)")
axes[2].set_title("SwiGLU: Gate Controls Signal Flow")
axes[2].set_xlabel("x")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Key insight: SiLU is smooth unlike ReLU's hard cutoff at 0.")
print("The gating mechanism lets the network LEARN which dimensions to keep.")