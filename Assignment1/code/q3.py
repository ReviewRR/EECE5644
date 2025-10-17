import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#read dataset
data = pd.read_csv("wine+quality/winequality-white.csv", sep=';')
X = data.drop(columns=['quality']).values
y = data['quality'].values


scaler = StandardScaler()
X_std = scaler.fit_transform(X)

classes, counts = np.unique(y, return_counts=True)
K = len(classes)
N = len(y)
priors = counts / N
print(f"Classes: {classes}")
print(f"Class priors: {priors}")

#Gaussian parameters

means = []
covs = []
lambda_reg = 1e-3

for c in classes:
    Xc = X_std[y == c]
    mu = np.mean(Xc, axis=0)
    cov = np.cov(Xc, rowvar=False)
    # Regularize covariance
    cov_reg = cov + lambda_reg * np.eye(cov.shape[0])
    means.append(mu)
    covs.append(cov_reg)

means = np.array(means)


# Minimum-Perror classification

likelihoods = np.zeros((N, K))
for i, c in enumerate(classes):
    rv = multivariate_normal(mean=means[i], cov=covs[i])
    likelihoods[:, i] = rv.pdf(X_std) * priors[i]

pred_labels = classes[np.argmax(likelihoods, axis=1)]

#confusion matrix

conf_mat = pd.crosstab(pred_labels, y,
                       rownames=['Predicted'], colnames=['True'],
                       normalize='columns')
print("\nConfusion Matrix P(D=i|L=j):")
print(conf_mat.round(3))

#Classification error

error_rate = np.mean(pred_labels != y)
print(f"\nEstimated classification error probability: {error_rate:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8,6))
for c in classes:
    idx = (y == c)
    plt.scatter(X_pca[idx,0], X_pca[idx,1], s=10, alpha=0.6, label=f'Class {c}')

plt.title('PCA projection of White-Wine Quality Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Quality', fontsize=8)
plt.tight_layout()
plt.show()


