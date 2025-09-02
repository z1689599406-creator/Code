import numpy as np
import matplotlib.pyplot as plt

# 生成数据（x1∈[0,1], x2∈[100,1000]）
np.random.seed(42)
X = np.random.rand(100, 2)
X[:, 1] = X[:, 1] * 900 + 100  # x2 ∈ [100,1000]
y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 10

# 未归一化的梯度下降
def gradient_descent(X, y, lr=0.01, epochs=100):
    w = np.zeros(2)
    losses = []
    for _ in range(epochs):
        y_pred = X.dot(w)
        gradient = X.T.dot(y_pred - y) / len(y)
        w -= lr * gradient
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
    return w, losses

w_raw, losses_raw = gradient_descent(X, y)

# Z-Score 归一化
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
w_norm, losses_norm = gradient_descent(X_norm, y)

# 绘制损失曲线
plt.plot(losses_raw, label="未归一化")
plt.plot(losses_norm, label="Z-Score归一化")
plt.legend()
plt.show()