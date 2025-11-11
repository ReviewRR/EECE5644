# MLP vs. MAP Classifier Comparison

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Define Gaussian Class-Conditional Distributions

np.random.seed(58)
torch.manual_seed(58)

C = 4           # number of classes
D = 3           # feature dimension
priors = [0.25] * 4

means = [
    np.array([0, 0, 0]),
    np.array([3, 3, 0]),
    np.array([0, 3, 3]),
    np.array([3, 0, 3])
]

'''Original Covariance set too close
covs = [
    np.diag([1.2, 1.0, 1.0]),
    np.diag([1.0, 1.2, 1.0]),
    np.diag([1.0, 1.0, 1.2]),
    np.diag([1.1, 1.1, 1.0])
]

'''



''' Expand the covariance to'''
covs = [
    np.diag([2.0, 1.8, 2.0]),
    np.diag([2.2, 2.0, 2.0]),
    np.diag([1.8, 2.0, 2.2]),
    np.diag([2.0, 2.2, 2.0])
]


# 2. Generate dataset

def sample_data(N):
    """Generate N samples from the 4-class Gaussian mixture"""
    X, y = [], []
    for i in range(C):
        n_i = int(N * priors[i])
        X_i = np.random.multivariate_normal(means[i], covs[i], n_i)
        y_i = np.full(n_i, i)
        X.append(X_i)
        y.append(y_i)
    return np.vstack(X), np.concatenate(y)


# Generate test set (fixed large set)
X_test, y_test = sample_data(100000)


# ----------------------------------------------------------
# 3. Theoretically Optimal MAP Classifier
# ----------------------------------------------------------
def map_classifier(X):
    """Compute MAP class decision using true Gaussian PDFs"""
    scores = []
    for i in range(C):
        diff = X - means[i]
        inv = np.linalg.inv(covs[i])
        det = np.linalg.det(covs[i])
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        score = np.exp(exponent) / np.sqrt((2 * np.pi) ** D * det)
        scores.append(score * priors[i])
    scores = np.array(scores).T
    return np.argmax(scores, axis=1)

y_map_pred = map_classifier(X_test)
P_error_opt = np.mean(y_map_pred != y_test)
print(f"Theoretical MAP Classifier Error ≈ {P_error_opt*100:.2f}%")

# ----------------------------------------------------------
# 4. Define the MLP Model
# ----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),          # smooth activation
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------
# 5. Training Utilities
# ----------------------------------------------------------
def train_mlp(X_train, y_train, hidden_dim, n_restarts=3, lr=1e-3, epochs=100):
    """Train MLP with multiple random restarts, pick best model"""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    best_model, best_loss = None, float("inf")
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_restarts):
        model = MLP(D, hidden_dim, C)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Record the final loss
        with torch.no_grad():
            model.eval()
            outputs = model(X_train_t)
            final_loss = criterion(outputs, y_train_t).item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model

    return best_model


def evaluate_model(model, X, y):
    """Return empirical classification error"""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_pred = torch.argmax(model(X_t), dim=1).numpy()
    return np.mean(y_pred != y)


# ----------------------------------------------------------
# 6. Cross-Validation to Select Hidden Layer Size (P)
# ----------------------------------------------------------
def cross_validate_hidden_dim(X, y, hidden_list, folds=10):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    avg_errors = []

    for h in hidden_list:
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = train_mlp(X_train, y_train, hidden_dim=h, epochs=50)
            err = evaluate_model(model, X_val, y_val)
            fold_errors.append(err)
        avg_errors.append(np.mean(fold_errors))
    best_h = hidden_list[np.argmin(avg_errors)]
    return best_h, avg_errors


# ----------------------------------------------------------
# 7. Run Experiments for Multiple Training Sizes
# ----------------------------------------------------------
train_sizes = [100, 500, 1000, 5000, 10000]
hidden_candidates = [2, 4, 8, 16, 32, 64]

results = {}

for N in train_sizes:
    print(f"\n=== Training with N = {N} samples ===")
    X_train, y_train = sample_data(N)

    # Cross-validation for model order
    best_h, cv_errors = cross_validate_hidden_dim(X_train, y_train, hidden_candidates)
    print(f"  → Selected hidden units: {best_h}")

    # Train final model on full training set
    final_model = train_mlp(X_train, y_train, hidden_dim=best_h, epochs=150)
    test_error = evaluate_model(final_model, X_test, y_test)

    results[N] = dict(best_h=best_h, P_error=test_error)
    print(f"  → Test set empirical P(error): {test_error*100:.2f}%")

# ----------------------------------------------------------
# 8. Plot Results
# ----------------------------------------------------------
Ns = list(results.keys())
P_errors = [results[n]['P_error'] for n in Ns]

plt.figure(figsize=(8,5))
plt.semilogx(Ns, P_errors, marker='o', label='MLP empirical P(error)')
plt.axhline(y=P_error_opt, color='r', linestyle='--', label='Theoretical MAP error')
plt.xlabel('Number of Training Samples (log scale)')
plt.ylabel('Empirical P(error)')
plt.title('MLP vs. Theoretical MAP Classifier')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 9. Display Summary Table
# ----------------------------------------------------------
print("\nSummary Results:")
print("N_train | Hidden P | Test Error (%)")
for N in Ns:
    print(f"{N:<8d} | {results[N]['best_h']:<9d} | {results[N]['P_error']*100:>8.2f}")
