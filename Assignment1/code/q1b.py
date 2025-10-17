import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

N = 10000
p0, p1 = 0.65, 0.35
m0 = np.array([-0.5, -0.5, -0.5])
C0 = np.array([[1, -0.5, 0.3], [-0.5, 1, -0.5], [0.3, -0.5, 1]])
m1 = np.array([1, 1, 1])
C1 = np.array([[1, 0.3, -0.2], [0.3, 1, 0.3], [-0.2, 0.3, 1]])

#Generate samples
u = np.random.rand(N)
labels = (u >= p0).astype(int)
N0, N1 = np.sum(labels == 0), np.sum(labels == 1)
r0 = np.random.multivariate_normal(m0, C0, N0)
r1 = np.random.multivariate_normal(m1, C1, N1)
X = np.vstack((r0, r1))
L_true = np.hstack((np.zeros(N0), np.ones(N1)))

# covariance
I3 = np.eye(3)

p_x_L0_nb = multivariate_normal.pdf(X, mean=m0, cov=I3)
p_x_L1_nb = multivariate_normal.pdf(X, mean=m1, cov=I3)

likelihood_ratio_nb = p_x_L1_nb / p_x_L0_nb

# roc
gamma_values = np.logspace(-8, 8, 900)
TPR, FPR, P_error = [], [], []

for gamma in gamma_values:
    decisions = (likelihood_ratio_nb > gamma).astype(int)
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
min_idx = np.argmin(P_error)

gamma_emp_nb = gamma_values[min_idx]
min_error_nb = P_error[min_idx]
gamma_theory = p0 / p1

# Plot
plt.figure(figsize=(6, 6))
plt.plot(FPR, TPR, label='Naive Bayes ROC (incorrect model)', color='orange')
plt.scatter(FPR[min_idx], TPR[min_idx], s=80, color='red',
            label=f"Min P(error) = {min_error_nb: .3f}")
plt.plot([0, 1], [0, 1], '--k')
plt.title('ROC Curve (Naive Bayes Classifier, Incorrect Covariance)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()
plt.show()

print(f"Theoretical γ*: {gamma_theory:.3f}")
print(f"Empirical γ* (NB): {gamma_emp_nb:.3f}")
print(f"Minimum P(error) (NB): {min_error_nb:.4f}")
