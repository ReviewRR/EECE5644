import numpy as np
from scipy.stats import multivariate_normal


np.random.seed(0)
N = 10000
priors = [0.25, 0.25, 0.25, 0.25]

mus = [
    np.array([0, 0]),
    np.array([3, 3]),
    np.array([-3, 3]),
    np.array([3, -3])
]

covs = [
    np.array([[1, 0.2], [0.2, 1]]),
    np.array([[1, -0.3], [-0.3, 1]]),
    np.array([[0.5, 0], [0, 0.5]]),
    np.array([[1.5, 0.4], [0.4, 1]])
]

#Generate samples
labels = np.random.choice([1, 2, 3, 4], size=N, p=priors)
X = np.zeros((N, 2))
for j in range(1, 5):
    idx = (labels == j)
    X[idx, :] = np.random.multivariate_normal(mus[j-1], covs[j-1], np.sum(idx))

# likelihoods
likelihoods = np.zeros((N, 4))
for j in range(4):
    likelihoods[:, j] = multivariate_normal.pdf(X, mean=mus[j], cov=covs[j]) * priors[j]

posterior = likelihoods / np.sum(likelihoods, axis=1, keepdims=True)

#Loss matrix
Lambda = np.array([
    [0, 10, 10, 100],
    [1, 0, 10, 100],
    [1, 1, 0, 100],
    [1, 1, 1, 0]
])


# R[i] = sum_l λ_il * P(L=l|x)
expected_risk = posterior @ Lambda.T   # shape: (N,4)

# ERM decision
pred_labels_erm = np.argmin(expected_risk, axis=1) + 1


total_risk = 0.0
for n in range(N):
    i = pred_labels_erm[n] - 1
    l = labels[n] - 1
    total_risk += Lambda[i, l]
avg_risk = total_risk / N
print(f"Estimated minimum expected risk (ERM): {avg_risk:.4f}")

#Compare with 0–1 loss MAP error
p_error_map = np.mean(pred_labels_erm != labels)
print(f"Empirical misclassification rate (for comparison): {p_error_map:.4f}")
