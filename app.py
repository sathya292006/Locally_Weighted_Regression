import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Housing.csv")

X = df["area"].values
y = df["price"].values

mean = X.mean()
std = X.std()

X_norm = (X - mean) / std
X_norm = np.vstack((np.ones(len(X_norm)), X_norm)).T

# LWR function
def lwr(X, y, query, tau):
    m = X.shape[0]
    weights = np.eye(m)

    for i in range(m):
        diff = query - X[i]
        weights[i, i] = np.exp(diff @ diff.T / (-2 * tau**2))

    theta = np.linalg.pinv(X.T @ weights @ X) @ (X.T @ weights @ y)
    return query @ theta

# Plot
def plot_graph(X, y, tau, input_val, pred):
    x_vals = X[:,1]
    x_sorted = np.sort(x_vals)
    y_pred = []

    for val in x_sorted:
        query = np.array([1, val])
        y_pred.append(lwr(X, y, query, tau))

    fig, ax = plt.subplots()
    ax.scatter(X[:,1], y)
    ax.plot(x_sorted, y_pred)
    ax.scatter(input_val, pred, marker='x', s=100)

    ax.set_xlabel("Normalized Area")
    ax.set_ylabel("Price")
    ax.set_title("House Price Prediction (LWR)")

    return fig

# UI
st.title("🏠 House Price Prediction using LWR")

tau = st.slider("Select Tau", 0.1, 2.0, 0.5)

area = st.number_input("Enter Area (sq ft)", 500, 10000, 2000)

if st.button("Predict"):
    area_norm = (area - mean) / std
    query = np.array([1, area_norm])

    pred = lwr(X_norm, y, query, tau)

    st.success(f"Predicted Price: {pred:.2f}")

    fig = plot_graph(X_norm, y, tau, area_norm, pred)
    st.pyplot(fig)