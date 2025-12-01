import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss
import warnings
warnings.filterwarnings("ignore")


# 1. Data generation
def generate_data(N):
    theta = np.random.uniform(-np.pi, np.pi, N)
    noise = np.random.randn(N, 2)  # sigma = 1
    labels = np.random.choice([-1, 1], N)
    r = np.where(labels == -1, 2, 4)
    x = np.vstack([r * np.cos(theta), r * np.sin(theta)]).T + noise
    return x, labels

np.random.seed(0)
X_train, y_train = generate_data(1000)


# 2. SVM Cross Validation + Curves
C_list = [0.1, 1, 10, 100]
gamma_list = [0.01, 0.1, 1, 10]
K = 5

kf = KFold(n_splits=K, shuffle=True, random_state=42)

svm_errors = {}  # key: gamma, value: list of errors under different C

for gamma in gamma_list:
    svm_errors[gamma] = []

    for C in C_list:
        fold_errors = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            clf = SVC(C=C, kernel="rbf", gamma=gamma)
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_val)
            fold_errors.append(zero_one_loss(y_val, y_pred))

        svm_errors[gamma].append(np.mean(fold_errors))
        print(f"SVM CV: gamma={gamma}, C={C}, error={np.mean(fold_errors):.4f}")


# ---- Plot SVM CV curves ----
plt.figure(figsize=(8,6))
for gamma in gamma_list:
    plt.plot(C_list, svm_errors[gamma], marker='o', label=f"gamma={gamma}")

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Validation Error")
plt.title("SVM Cross-Validation Error Curves")
plt.legend()
plt.grid(True)
plt.show()


# 3. MLP Cross Validation + Curves

# Quadratic feature mapping
def quadratic_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.column_stack([x1, x2, x1**2, x2**2, x1*x2])

Xq_train = quadratic_features(X_train)

hidden_list = [5, 10, 20, 50, 100]
mlp_errors = []

for h in hidden_list:
    fold_errors = []

    for train_idx, val_idx in kf.split(Xq_train):
        X_tr, X_val = Xq_train[train_idx], Xq_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        clf = MLPClassifier(
            hidden_layer_sizes=(h,),
            activation="relu",
            max_iter=500,
            random_state=0
        )
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_val)
        fold_errors.append(zero_one_loss(y_val, y_pred))

    mlp_errors.append(np.mean(fold_errors))
    print(f"MLP CV: hidden={h}, error={np.mean(fold_errors):.4f}")


# ---- Plot MLP CV curve ----
plt.figure(figsize=(8,6))
plt.plot(hidden_list, mlp_errors, marker='o')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Validation Error")
plt.title("MLP Cross-Validation Error Curve")
plt.grid(True)
plt.show()


# Train final SVM with best hyperparameters
# Find best SVM params from CV results
best_svm_error = 1e9
best_C = None
best_gamma = None

for gamma in gamma_list:
    for C, err in zip(C_list, svm_errors[gamma]):
        if err < best_svm_error:
            best_svm_error = err
            best_C = C
            best_gamma = gamma

print("\nBest SVM parameters: C=", best_C, " gamma=", best_gamma)

# Train final SVM
svm_clf = SVC(C=best_C, kernel="rbf", gamma=best_gamma)
svm_clf.fit(X_train, y_train)

# Test error
X_test, y_test = generate_data(10000)
svm_test_error = zero_one_loss(y_test, svm_clf.predict(X_test))
print("SVM Test Error =", svm_test_error)


# Train final MLP with best hidden units
best_hidden = hidden_list[np.argmin(mlp_errors)]
print("\nBest MLP hidden units =", best_hidden)

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(best_hidden,),
    activation="relu",
    max_iter=500,
    random_state=0
)
mlp_clf.fit(Xq_train, y_train)

Xq_test = quadratic_features(X_test)
mlp_test_error = zero_one_loss(y_test, mlp_clf.predict(Xq_test))
print("MLP Test Error =", mlp_test_error)


# Plot decision boundary
def plot_boundary(clf, transform, title):
    xx, yy = np.meshgrid(
        np.linspace(-8, 8, 400),
        np.linspace(-8, 8, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid2 = transform(grid)
    Z = clf.predict(grid2).reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1,0,1])
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', s=5)
    plt.title(title)
    plt.show()


plot_boundary(svm_clf, lambda x:x, "SVM Decision Boundary")
plot_boundary(mlp_clf, quadratic_features, "MLP Decision Boundary")
