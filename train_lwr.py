import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Housing.csv")

# Use only area and price
X = df["area"].values
y = df["price"].values

# Normalize (important for LWR)
X = (X - X.mean()) / X.std()

# Add bias
X = np.vstack((np.ones(len(X)), X)).T

# -------- LWR FUNCTION --------
def lwr(X, y, query, tau):
    m = X.shape[0]
    weights = np.eye(m)

    for i in range(m):
        diff = query - X[i]
        weights[i, i] = np.exp(diff @ diff.T / (-2 * tau**2))

    theta = np.linalg.pinv(X.T @ weights @ X) @ (X.T @ weights @ y)
    return query @ theta

# -------- PLOT --------
def plot_lwr(X, y, tau):
    x_vals = X[:,1]
    x_sorted = np.sort(x_vals)
    y_pred = []

    for val in x_sorted:
        query = np.array([1, val])
        y_pred.append(lwr(X, y, query, tau))

    return x_sorted, y_pred

# -------- MAIN --------
tau = 0.5

x_curve, y_curve = plot_lwr(X, y, tau)

plt.scatter(X[:,1], y, label="Data")
plt.plot(x_curve, y_curve, color='red', label="LWR Curve")
plt.xlabel("Normalized Area")
plt.ylabel("Price")
plt.title("LWR - House Price Prediction")
plt.legend()
plt.show()

# Prediction
input_area = float(input("Enter area (sq ft): "))

# Normalize input
input_norm = (input_area - df["area"].mean()) / df["area"].std()

query = np.array([1, input_norm])
prediction = lwr(X, y, query, tau)

print(f"Predicted House Price: {prediction:.2f}")