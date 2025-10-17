import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(0)
N = 10000
priors = [0.25, 0.25, 0.25, 0.25]

#Define Gaussian parameters
mus = [np.array([0, 0]),
       np.array([3, 3]),
       np.array([-3, 3]),
       np.array([3, -3])]

covs = [np.array([[1, 0.2], [0.2, 1]]),
        np.array([[1, -0.3], [-0.3, 1]]),
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([[1.5, 0.4], [0.4, 1]])]

# class labels according to priors
labels = np.random.choice([1, 2, 3, 4], size=N, p=priors)

#Sample
X = np.zeros((N, 2))
for j in range(1, 5):
    idx = (labels == j)
    nj = np.sum(idx)
    X[idx, :] = np.random.multivariate_normal(mus[j-1], covs[j-1], nj)

# MAP classification likelihood
likelihoods = np.zeros((N, 4))
for j in range(4):
    likelihoods[:, j] = multivariate_normal.pdf(X, mean=mus[j], cov=covs[j]) * priors[j]

pred_labels = np.argmax(likelihoods, axis=1) + 1

#classification error
p_error = np.mean(pred_labels != labels)
print(f"Empirical probability of error (MAP): {p_error:.4f}")

#Plot
plt.figure(figsize=(7,6))
colors = ['b','g','r','orange']
for j in range(4):
    plt.scatter(X[labels==j+1,0], X[labels==j+1,1], s=10,
                alpha=0.4, color=colors[j], label=f"Class {j+1}")
plt.xlabel("x₁"); plt.ylabel("x₂")
plt.legend(); plt.axis('equal')
plt.title("Selected Gaussian PDF Samples")
plt.show()

## question 2
likelihoods = np.zeros((len(X), 4))
for j in range(4):
    likelihoods[:, j] = multivariate_normal.pdf(X, mean=mus[j], cov=covs[j]) * priors[j]


pred_labels = np.argmax(likelihoods, axis=1) + 1

# Confusion matrix
K = 4
conf_mat = np.zeros((K, K))
for j_true in range(1, K+1):
    idx = (labels == j_true)
    Nj = np.sum(idx)
    for i_dec in range(1, K+1):
        conf_mat[i_dec-1, j_true-1] = np.sum(pred_labels[idx] == i_dec) / Nj

print("Confusion matrix  P(D=i | L=j):\n")
print(np.round(conf_mat, 4))

'''
question 3
'''
# define shapes for 4 classes
markers = ['o', '^', 's', 'D']
plt.figure(figsize=(8,7))

for j in range(1,5):
    idx_true = (labels == j)
    idx_correct = idx_true & (pred_labels == j)
    idx_wrong = idx_true & (pred_labels != j)

    # plot correctly classified samples (green)
    plt.scatter(X[idx_correct, 0], X[idx_correct, 1],
                marker=markers[j - 1], color='green', edgecolor='k',
                s=20, alpha=0.6, label=f'Class {j} (correct)')
    plt.scatter(X[idx_wrong, 0], X[idx_wrong, 1],
                marker=markers[j - 1], color='red', edgecolor='k',
                s=30, alpha=0.9, label=f'Class {j} (wrong)')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('0-1 Loss Classification Correctness')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)


plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()

plt.show()