# ============================================================
# EECE5644 HW2  |  Part 1 + Part 2 (Linear / Quadratic Logistic)
# Author: Sharon
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

np.random.seed(0)

# ============================================================
# 1. Data generation (two-class GMM)
# ============================================================
priors = [0.6, 0.4]
C = np.array([[0.75, 0], [0, 1.25]])
m01, m02 = np.array([-0.9, -1.1]), np.array([0.8, 0.75])
m11, m12 = np.array([-1.1, 0.9]), np.array([0.9, -0.75])
mu0_list = np.column_stack((m01, m02))
mu1_list = np.column_stack((m11, m12))
alpha = [0.5, 0.5]

def rand_gmm(N, alpha, mu_list, Sigma_list):
    d = mu_list.shape[0]
    x = np.zeros((d, N))
    u = np.random.rand(N)
    cum = np.hstack(([0], np.cumsum(alpha)))
    for k in range(len(alpha)):
        idx = (u > cum[k]) & (u <= cum[k + 1])
        x[:, idx] = np.random.multivariate_normal(mu_list[:, k], Sigma_list[k], np.sum(idx)).T
    return x

def generate_dataset(N):
    labels = (np.random.rand(N) >= priors[0]).astype(int)
    x = np.zeros((2, N))
    n0, n1 = np.sum(labels == 0), np.sum(labels == 1)
    x[:, labels == 0] = rand_gmm(n0, alpha, mu0_list, [C, C])
    x[:, labels == 1] = rand_gmm(n1, alpha, mu1_list, [C, C])
    return x.T, labels

datasets = {}
for N in [50, 500, 5000, 10000]:
    datasets[f"D{N}"] = generate_dataset(N)

# ============================================================
# 2. Scatter plots for different sample sizes
# ============================================================
Ns = [50, 500, 5000, 10000]
plt.figure(figsize=(10, 8))
for i, N in enumerate(Ns, 1):
    X, y = datasets[f"D{N}"]
    plt.subplot(2, 2, i)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=10, label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=10, label="Class 1")
    plt.title(f"{N} samples")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 3. Theoretical Bayes classifier (Part 1)
# ============================================================
def eval_gmm(x, alpha, mu_list, Sigma_list):
    pdf = np.zeros(x.shape[0])
    for k in range(len(alpha)):
        pdf += alpha[k] * multivariate_normal.pdf(x, mean=mu_list[:, k], cov=Sigma_list[k])
    return pdf

Xv, yv = datasets["D10000"]
px0 = eval_gmm(Xv, alpha, mu0_list, [C, C])
px1 = eval_gmm(Xv, alpha, mu1_list, [C, C])
disc = np.log(px1) - np.log(px0)

# Sweep threshold for ROC
thr = np.sort(disc)
FP, TP, Err = [], [], []
N0, N1 = np.sum(yv == 0), np.sum(yv == 1)
for g in thr:
    decide = disc >= g
    FP.append(np.sum(decide & (yv == 0)) / N0)
    TP.append(np.sum(decide & (yv == 1)) / N1)
    Err.append(FP[-1] * priors[0] + (1 - TP[-1]) * priors[1])
FP, TP, Err = np.array(FP), np.array(TP), np.array(Err)
min_idx = np.argmin(Err)
log_gamma_ideal = np.log(priors[0] / priors[1])

# Plot ROC
plt.figure(figsize=(6, 5))
plt.plot(FP, TP, 'b-', label="ROC Curve")
plt.plot(FP[min_idx], TP[min_idx], 'ro', label="Empirical min-P(error)")
plt.plot(FP[np.argmin(np.abs(thr - log_gamma_ideal))],
         TP[np.argmin(np.abs(thr - log_gamma_ideal))],
         'k+', markersize=10, label="Ideal threshold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bayes Optimal ROC Curve")
plt.legend(); plt.grid(True)
plt.show()

print(f"Ideal log(gamma)={log_gamma_ideal:.3f}")
print(f"Empirical min-P(error)={Err[min_idx]*100:.2f}% at log(gamma)={thr[min_idx]:.3f}")

# ============================================================
# 4. Logistic regression models (Part 2a, 2b)
# ============================================================
def sigmoid(z): return 1 / (1 + np.exp(-z))
def nll(theta, X, y):
    h = sigmoid(X @ theta)
    eps = 1e-9
    return -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))

def train_logistic(X, y, quadratic=False):
    if quadratic:
        x1, x2 = X[:, 0], X[:, 1]
        X_aug = np.column_stack([np.ones(len(X)), x1, x2, x1**2, x1*x2, x2**2])
    else:
        X_aug = np.column_stack([np.ones(len(X)), X])
    theta0 = np.zeros(X_aug.shape[1])
    res = minimize(lambda t: nll(t, X_aug, y), theta0)
    return res.x

def predict_logistic(theta, X, quadratic=False):
    if quadratic:
        x1, x2 = X[:, 0], X[:, 1]
        X_aug = np.column_stack([np.ones(len(X)), x1, x2, x1**2, x1*x2, x2**2])
    else:
        X_aug = np.column_stack([np.ones(len(X)), X])
    probs = sigmoid(X_aug @ theta)
    return (probs >= 0.5).astype(int), probs

def plot_boundary(theta, X, y, quadratic=False, title=""):
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, probs = predict_logistic(theta, grid, quadratic)
    zz = probs.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=["#AAD", "#FBB"], alpha=0.3)
    plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=10, label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=10, label="Class 1")
    plt.title(title)
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
    plt.legend(); plt.grid(True)
    plt.show()

# ============================================================
# 5. Train and evaluate models on D10K_validate
# ============================================================
X_val, y_val = datasets["D10000"]

for N in [50, 500, 5000]:
    X_train, y_train = datasets[f"D{N}"]

    # Linear logistic
    theta_lin = train_logistic(X_train, y_train, quadratic=False)
    y_pred_lin, _ = predict_logistic(theta_lin, X_val, quadratic=False)
    err_lin = np.mean(y_pred_lin != y_val)
    print(f"[Linear Logistic N={N}] P(error)={err_lin*100:.2f}%")
    plot_boundary(theta_lin, X_train, y_train, quadratic=False,
                  title=f"Linear Logistic Decision Boundary (N={N})")

    # Quadratic logistic
    theta_quad = train_logistic(X_train, y_train, quadratic=True)
    y_pred_quad, _ = predict_logistic(theta_quad, X_val, quadratic=True)
    err_quad = np.mean(y_pred_quad != y_val)
    print(f"[Quadratic Logistic N={N}] P(error)={err_quad*100:.2f}%")
    plot_boundary(theta_quad, X_train, y_train, quadratic=True,
                  title=f"Quadratic Logistic Decision Boundary (N={N})")
