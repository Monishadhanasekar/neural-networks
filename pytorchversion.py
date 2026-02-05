import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------- DATA ----------------
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# ---------------- MODEL ----------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 2)   # 1 input → 2 neurons
        self.output = nn.Linear(2, 1)   # 2 neurons → 1 output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = SimpleNN()

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5000
losses = []

# ---------------- TRAINING LOOP ----------------
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # 🔥 Backprop happens automatically
    optimizer.step()       # Update weights

# ---------------- RESULTS ----------------
print("Final Loss:", loss.item())

test = torch.tensor([[5.0]])
prediction = model(test)
print("Prediction for x=5:", prediction.item())

# ---------------- LOSS GRAPH ----------------
plt.plot(losses)
plt.title("Loss Decreasing")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
