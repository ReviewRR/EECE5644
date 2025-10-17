import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


train_X = np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt")
train_y = np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt", dtype=int)
test_X = np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt")
test_y = np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt", dtype=int)

# Combine the two datasets
X = np.vstack((train_X, test_X))
y = np.hstack((train_y, test_y))

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features.")

# map the label
activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Convert numeric labels to strings
y_labels = np.array([activity_map[i] for i in y])
activity_names = list(activity_map.values())

print("Activity labels:", activity_names)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

classes, counts = np.unique(y_labels, return_counts=True)
K = len(classes)
N = len(y_labels)
priors = counts / N
lambda_reg = 1e-3

means, covs = [], []
for c in classes:
    Xc = X_std[y_labels == c]
    mu = np.mean(Xc, axis=0)
    cov = np.cov(Xc, rowvar=False)
    cov_reg = cov + lambda_reg * np.eye(cov.shape[0])
    means.append(mu)
    covs.append(cov_reg)

means = np.array(means)

#MAP Rule
likelihoods = np.zeros((N, K))
for i, c in enumerate(classes):
    rv = multivariate_normal(mean=means[i], cov=covs[i])
    likelihoods[:, i] = rv.pdf(X_std) * priors[i]

pred_labels = classes[np.argmax(likelihoods, axis=1)]

#Confusion matrix
conf_mat = pd.crosstab(pred_labels, y_labels,
                       rownames=['Predicted'], colnames=['True'],
                       normalize='columns')
print("\nConfusion Matrix P(D=i|L=j):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(conf_mat.round(3))

# Classification error

error_rate = np.mean(pred_labels != y_labels)
print(f"\nEstimated classification error probability: {error_rate:.4f}")

#PCA visualization

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8,6))
for c in classes:
    idx = (y_labels == c)
    plt.scatter(X_pca[idx,0], X_pca[idx,1], s=8, alpha=0.6, label=c)
plt.title('PCA Projection of HAR Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Activity', fontsize=8)
plt.tight_layout()
plt.show()
