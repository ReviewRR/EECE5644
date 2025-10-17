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

# Generate the dataset
u = np.random.rand(N)
labels = np.zeros(N, dtype=int)   # L=0 or 1
labels[u >= p0] = 1               # allocate the tag base on probability

N0 = np.sum(labels == 0)
N1 = np.sum(labels == 1)

# Generate the sample
r0 = np.random.multivariate_normal(m0, C0, N0)
r1 = np.random.multivariate_normal(m1, C1, N1)

X = np.vstack((r0, r1))
L_true = np.hstack((np.zeros(N0), np.ones(N1)))

p_x_L0 = multivariate_normal.pdf(X, mean=m0, cov=C0)
p_x_L1 = multivariate_normal.pdf(X, mean=m1, cov=C1)


# 0–1 loss： gamma = P(L=0) / P(L=1)
gamma = p0 / p1
likelihood_ratio = p_x_L1 / p_x_L0

L_pred = (likelihood_ratio > gamma).astype(int)

#error rate
error_rate = np.mean(L_pred != L_true)
print(f"Classification error rate: {error_rate:.4f}")

#compute the classification performance
num_errors = np.sum(L_pred != L_true)
error_rate = num_errors / N
print(f"Total classification errors: {num_errors}/{N}")
print(f"Empirical classification error rate: {error_rate:.4f}")

#plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(r0[:,0], r0[:,1], r0[:,2], c='blue', s=10, alpha=0.5, label='Class 0')
ax.scatter(r1[:,0], r1[:,1], r1[:,2], c='red', s=10, alpha=0.5, label='Class 1')

ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('X3')
ax.set_title('Generated Data')
ax.legend()
plt.show()