# 1 input -> 2 hidden neurons -> 1 output

import numpy as np
import matplotlib.pyplot as plt

# ---------------- DATA ----------------
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[2], [4], [6], [8]], dtype=float)

np.random.seed(0)

# -------------- NETWORK SIZE --------------
input_size = 1
hidden_size = 2   # 🔥 ONLY 2 HIDDEN NEURONS
output_size = 1

# -------------- WEIGHTS ----------------
W1 = np.random.randn(input_size, hidden_size)   # (1x2)
b1 = np.zeros((1, hidden_size))                 # (1x2)

W2 = np.random.randn(hidden_size, output_size)  # (2x1)
b2 = np.zeros((1, output_size))                 # (1x1)

learning_rate = 0.01
epochs = 2000
losses = []

# -------------- ACTIVATION --------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# -------------- TRAINING LOOP --------------
for epoch in range(epochs):

    # ---- Forward Pass ----
    z1 = X.dot(W1) + b1      # (4x2)
    a1 = relu(z1)            # (4x2)

    z2 = a1.dot(W2) + b2     # (4x1)
    y_pred = z2              # (4x1)

    # ---- Loss ----
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # ---- Backprop ----
    dloss = 2 * (y_pred - y) / y.size  # (4x1)

    dW2 = a1.T.dot(dloss)              # (2x1)
    db2 = np.sum(dloss, axis=0, keepdims=True)

    da1 = dloss.dot(W2.T)              # (4x2)
    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T.dot(dz1)                 # (1x2)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ---- Update ----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# -------------- RESULTS --------------
print("Final Loss:", loss)
print("W1:", W1)
print("W2:", W2)

# Test
test = np.array([[5]])
prediction = relu(test.dot(W1) + b1).dot(W2) + b2
print("Prediction for x=5:", prediction)

# -------------- LOSS GRAPH --------------
plt.plot(losses)
plt.title("Loss Decreasing")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


