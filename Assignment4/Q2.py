import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
#from skimage.io import imread
from sklearn.preprocessing import MinMaxScaler


# ========== Step 1: read image ==========
img = mpimg.imread("197017.jpg")
H, W, C = img.shape
print("Original image size:", img.shape)

# ========== Step 2: create 5-D feature vectors ==========
rows, cols = np.indices((H, W))
rows = rows.reshape(-1, 1)
cols = cols.reshape(-1, 1)
rgb = img.reshape(-1, 3)

X_raw = np.hstack([rows, cols, rgb])  # shape (N,5)

# ========== Step 3: normalize each feature to [0,1] ==========
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# ========== Step 4: K-fold CV to pick best K ==========
possible_K = [2, 3, 4, 5, 6, 8,10,13,15,20]
kf = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_score(K):
    ll = []
    for train_idx, val_idx in kf.split(X):
        gm = GaussianMixture(n_components=K, covariance_type='full', max_iter=200)
        gm.fit(X[train_idx])
        ll.append(gm.score(X[val_idx]))  # avg log-likelihood per sample
    return np.mean(ll)

scores = {}
for K in possible_K:
    scores[K] = cv_score(K)
    print(f"K={K}, avg validation log-likelihood={scores[K]:.3f}")

best_K = max(scores, key=scores.get)
print("Best K =", best_K)

# ========== Step 5: Fit GMM with the best K ==========
best_gmm = GaussianMixture(n_components=best_K, covariance_type='full', max_iter=200)
best_gmm.fit(X)

# ========== Step 6: Predict labels ==========
labels = best_gmm.predict(X)
label_map = labels.reshape(H, W)

# ========== Step 7: Convert label image to grayscale 0-255 ==========
seg_vis = (label_map / label_map.max() * 255).astype(np.uint8)

# ========== Step 8: Show side-by-side ==========
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(seg_vis, cmap='gray')
plt.title(f"GMM Segmentation (K={best_K})")
plt.axis("off")

plt.show()
