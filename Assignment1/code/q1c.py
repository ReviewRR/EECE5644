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
r0 = np.random.multivariate_normal(m0, C0, np.sum(labels==0))
r1 = np.random.multivariate_normal(m1, C1, np.sum(labels==1))
X = np.vstack((r0, r1))
L_true = np.hstack((np.zeros(len(r0)), np.ones(len(r1))))

#sample means and covariances
mu0_hat = np.mean(X[L_true==0], axis=0)
mu1_hat = np.mean(X[L_true==1], axis=0)
S0_hat = np.cov(X[L_true==0].T)
S1_hat = np.cov(X[L_true==1].T)

#Fisher LDA weight vector
Sw = S0_hat + S1_hat      # within-class scatter
wLDA = np.linalg.inv(Sw) @ (mu1_hat - mu0_hat)
wLDA = wLDA / np.linalg.norm(wLDA)  # optional normalization


y = X @ wLDA

#Sweep thresholds
tau_values = np.linspace(y.min(), y.max(), 400)
TPR, FPR, P_error = [], [], []

for tau in tau_values:
    decisions = (y > tau).astype(int)
    TP = np.sum((decisions==1)&(L_true==1))
    FN = np.sum((decisions==0)&(L_true==1))
    FP = np.sum((decisions==1)&(L_true==0))
    TN = np.sum((decisions==0)&(L_true==0))
    tpr = TP/(TP+FN)
    fpr = FP/(FP+TN)
    perror = fpr*p0 + (1-tpr)*p1
    TPR.append(tpr)
    FPR.append(fpr)
    P_error.append(perror)

TPR, FPR, P_error = map(np.array, (TPR, FPR, P_error))
min_idx = np.argmin(P_error)
tau_opt = tau_values[min_idx]
Pmin = P_error[min_idx]

#Plot
plt.figure(figsize=(6,6))
plt.plot(FPR, TPR, label='LDA ROC', color='green')
plt.scatter(FPR[min_idx], TPR[min_idx], s=80, color='red',
            label=f"Min P(error) = {Pmin: .4f}")
plt.plot([0,1],[0,1],'--k')
plt.title('ROC Curve – Fisher LDA Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal τ*: {tau_opt:.3f}")
print(f"Minimum P(error): {Pmin:.4f}")
