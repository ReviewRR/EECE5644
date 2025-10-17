import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# ---------- Parameters ----------
N = 10000
p0 = 0.65
p1 = 0.35

m0 = np.array([-0.5, -0.5, -0.5])
C0 = np.array([[1, -0.5, 0.3],
               [-0.5, 1, -0.5],
               [0.3, -0.5, 1]])
m1 = np.array([1, 1, 1])
C1 = np.array([[1, 0.3, -0.2],
               [0.3, 1, 0.3],
               [-0.2, 0.3, 1]])

# Generate samples
u = np.random.rand(N)
labels = (u >= p0).astype(int)
N0 = np.sum(labels == 0)
N1 = np.sum(labels == 1)
r0 = np.random.multivariate_normal(m0, C0, N0)
r1 = np.random.multivariate_normal(m1, C1, N1)
X = np.vstack((r0, r1))
L_true = np.hstack((np.zeros(N0), np.ones(N1)))

#Likelihood ratio
p_x_L0 = multivariate_normal.pdf(X, mean=m0, cov=C0)
p_x_L1 = multivariate_normal.pdf(X, mean=m1, cov=C1)
likelihood_ratio = p_x_L1 / p_x_L0

#Sweep gamma
gamma_values = np.logspace(-8, 8, 900)
TPR, FPR, P_error = [], [], []

for gamma in gamma_values:
    decisions = (likelihood_ratio > gamma).astype(int)
    TP = np.sum((decisions == 1) & (L_true == 1))
    FN = np.sum((decisions == 0) & (L_true == 1))
    FP = np.sum((decisions == 1) & (L_true == 0))
    TN = np.sum((decisions == 0) & (L_true == 0))

    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    perror = fpr * p0 + (1 - tpr) * p1

    TPR.append(tpr)
    FPR.append(fpr)
    P_error.append(perror)

TPR, FPR, P_error = map(np.array, (TPR, FPR, P_error))

#empirical optimum
min_idx = np.argmin(P_error)
gamma_empirical = gamma_values[min_idx]
min_error = P_error[min_idx]

gamma_theory = p0 / p1  # = 1.857

#Plot
plt.figure(figsize=(6, 6))
plt.plot(FPR, TPR, label='ROC Curve', color='blue')
plt.scatter(FPR[min_idx], TPR[min_idx], color='red', s=80,
            label=f"Min P(error) = {min_error: .3f}")
plt.plot([0, 1], [0, 1], '--k')
plt.title('ROC Curve with Minimum P(error) Point')
plt.xlabel('False Positive Rate P(D=1|L=0)')
plt.ylabel('True Positive Rate P(D=1|L=1)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Theoretical γ*: {gamma_theory:.3f}")
print(f"Empirical γ*: {gamma_empirical:.3f}")
print(f"Minimum P(error): {min_error:.4f}")
