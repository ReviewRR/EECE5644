import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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

#Generate the sample
u = np.random.rand(N)
labels = np.zeros(N)
labels[u >= p0] = 1

N0 = np.sum(labels == 0)
N1 = np.sum(labels == 1)

r0 = np.random.multivariate_normal(m0, C0, N0)
r1 = np.random.multivariate_normal(m1, C1, N1)
X = np.vstack((r0, r1))
L_true = np.hstack((np.zeros(N0), np.ones(N1)))

p_x_L0 = multivariate_normal.pdf(X, mean=m0, cov=C0)
p_x_L1 = multivariate_normal.pdf(X, mean=m1, cov=C1)
likelihood_ratio = p_x_L1 / p_x_L0

gamma_values = np.logspace(-8, 8, 900)
TPR = []  # True Positive Rate = P(D=1|L=1)
FPR = []  # False Positive Rate = P(D=1|L=0)

for gamma in gamma_values:
    decisions = (likelihood_ratio > gamma).astype(int)
    TP = np.sum((decisions == 1) & (L_true == 1))
    FN = np.sum((decisions == 0) & (L_true == 1))
    FP = np.sum((decisions == 1) & (L_true == 0))
    TN = np.sum((decisions == 0) & (L_true == 0))

    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    TPR.append(tpr)
    FPR.append(fpr)

TPR = np.array(TPR)
FPR = np.array(FPR)

# roc curve
plt.figure(figsize=(6,6))
plt.plot(FPR, TPR, '-b', label='ROC curve')
plt.plot([0,1], [0,1], '--k', label='Random guess line')
plt.title('ROC Curve of Minimum Expected Risk Classifier')
plt.xlabel('False Positive Rate P(D=1|L=0)')
plt.ylabel('True Positive Rate P(D=1|L=1)')
plt.grid(True)
plt.legend()
plt.axis('square')
plt.show()

for g, t, f in zip(gamma_values[::50], TPR[::50], FPR[::50]):
    print(f"gamma={g:.3f}, TPR={t:.3f}, FPR={f:.3f}")
