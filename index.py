#single linear neuron

import numpy as np

# Training data (y = 2x)
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[2], [4], [6], [8]], dtype=float)

# Random start
np.random.seed(0)
w = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.01
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    y_pred = X * w + b
    
    # Error
    error = y_pred - y
    
    # Gradients
    dw = np.mean(error * X)
    db = np.mean(error)
    
    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

print("Final weight (w):", w)
print("Final bias (b):", b)

# Test prediction
test = np.array([[5]])
prediction = test * w + b
print("Prediction for x=5:", prediction)
