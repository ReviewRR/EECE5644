import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from hw2q2 import hw2q2

# ---------------- Data ----------------
xTrain, yTrain, xVal, yVal = hw2q2()
xTrain, yTrain = xTrain.T, yTrain.flatten()
xVal, yVal = xVal.T, yVal.flatten()

# ---------------- Feature scaling ----------------
x_mean, x_std = xTrain.mean(axis=0), xTrain.std(axis=0)
xTrain_n = (xTrain - x_mean) / x_std
xVal_n   = (xVal   - x_mean) / x_std

# ---------------- Cubic feature map ----------------
def cubic_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([
        np.ones(len(X)), x1, x2,
        x1**2, x1*x2, x2**2,
        x1**3, x1**2*x2, x1*x2**2, x2**3
    ])

Phi_train = cubic_features(xTrain_n)
Phi_val   = cubic_features(xVal_n)

# ---------------- Estimators ----------------
def ml_estimator(Phi, y):
    return np.linalg.pinv(Phi) @ y

def map_estimator(Phi, y, sigma2, gamma):
    lam = sigma2 / gamma
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    return np.linalg.solve(A, Phi.T @ y)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# ---------------- ML baseline ----------------
sigma2 = np.var(yTrain - Phi_train @ np.linalg.lstsq(Phi_train, yTrain, rcond=None)[0])
w_ml = ml_estimator(Phi_train, yTrain)
mse_ml = mse(yVal, Phi_val @ w_ml)
print(f"ML Validation MSE = {mse_ml:.4f}")

# ---------------- MAP sweep ----------------
gamma_vals = np.logspace(-1, 3, 40)
mse_map = []

for g in gamma_vals:
    w_map = map_estimator(Phi_train, yTrain, sigma2, g)
    mse_g = mse(yVal, Phi_val @ w_map)
    mse_map.append(mse_g)

mse_map = np.array(mse_map)
best_idx = np.nanargmin(mse_map)
print(f"Best MAP MSE = {mse_map[best_idx]:.4f} at gamma = {gamma_vals[best_idx]:.3f}")

# ---------------- Plot ----------------
plt.close('all')
plt.figure(figsize=(6, 4))
plt.semilogx(gamma_vals, mse_map, 'bo-', label='MAP Validation MSE')
plt.axhline(mse_ml, color='r', linestyle='--', label='ML (γ→∞)')
plt.axvline(gamma_vals[best_idx], color='g', linestyle=':', label=f'Best γ={gamma_vals[best_idx]:.3f}')
plt.xlabel(r'$\gamma$ (prior variance)')
plt.ylabel('Mean Squared Error')
plt.title('Validation MSE vs Prior Variance γ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)
