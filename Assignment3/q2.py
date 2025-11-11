
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import Counter

# Generate N samples from a 4-component 2D GMM
def generate_true_gmm(N, seed=None):

    if seed is not None:
        np.random.seed(seed)
    pis = [0.2, 0.3, 0.3, 0.2]
    mus = np.array([[0, 0], [1.2, 1.2], [5, 0], [-3, 4]])
    covs = np.array([
        [[1, 0.3], [0.3, 1]],
        [[1, 0.2], [0.2, 1]],
        [[0.5, 0], [0, 0.8]],
        [[0.6, -0.1], [-0.1, 0.6]]
    ])
    z = np.random.choice(4, size=N, p=pis)
    X = np.vstack([np.random.multivariate_normal(mus[i], covs[i], 1) for i in z])
    return X

# Compute mean validation log-likelihood for K=1 - 10
def crossval_gmm(X, Kmax=10, nfold=10):

    kf = KFold(n_splits=nfold, shuffle=True)
    avg_loglik = []
    for k in range(1, Kmax + 1):
        fold_ll = []
        for train_idx, val_idx in kf.split(X):
            # skip if not enough samples for current K
            if len(train_idx) < k:
                continue
            gm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
            gm.fit(X[train_idx])
            fold_ll.append(gm.score(X[val_idx]))
        if len(fold_ll) > 0:
            avg_loglik.append(np.mean(fold_ll))
        else:
            avg_loglik.append(np.nan)
    return np.array(avg_loglik)

# Repeat cross-validation for multiple dataset sizes
def run_experiment(N_list=[10, 100, 1000], repeats=100, Kmax=10):
    results = {N: [] for N in N_list}
    ll_records = {N: [] for N in N_list}
    for N in N_list:
        for r in range(repeats):
            X = generate_true_gmm(N, seed=r)
            avg_ll = crossval_gmm(X, Kmax=min(Kmax, len(X)-1))
            best_k = np.nanargmax(avg_ll) + 1
            results[N].append(best_k)
            ll_records[N].append(avg_ll)
    return results, ll_records

# Print frequency table and plot histograms + log-likelihood trends
def summarize_results(results, ll_records, Kmax=10):
    for N, Ks in results.items():
        print(f"\n=== Dataset size: N={N} ===")
        counts = Counter(Ks)
        total = sum(counts.values())
        for k in range(1, Kmax + 1):
            freq = counts[k] / total * 100
            print(f"K={k:<2d}  ->  {freq:5.1f}%")

        # Histogram of selected Ks
        plt.figure(figsize=(5, 4))
        plt.hist(Ks, bins=np.arange(1, Kmax + 2) - 0.5, edgecolor='black')
        plt.title(f'Model Order Selection (N={N})')
        plt.xlabel('Selected Model Order K')
        plt.ylabel('Frequency (out of 100)')
        plt.xticks(range(1, Kmax + 1))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Mean log-likelihood vs. K
        ll_array = np.array(ll_records[N])
        # Handle potential NaNs or variable-length arrays
        valid_cols = np.isfinite(np.nanmean(ll_array, axis=0))
        mean_ll = np.nanmean(ll_array[:, valid_cols], axis=0)
        std_ll = np.nanstd(ll_array[:, valid_cols], axis=0)
        K_valid = len(mean_ll)

        plt.figure(figsize=(5.5, 4))
        plt.errorbar(range(1, K_valid + 1), mean_ll, yerr=std_ll, fmt='-o', capsize=4)
        plt.title(f'Cross-Validated Log-Likelihood vs. Model Order (N={N})')
        plt.xlabel('Number of Components (K)')
        plt.ylabel('Mean Log-Likelihood (10-fold CV)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    results, ll_records = run_experiment(N_list=[10, 100, 1000], repeats=100, Kmax=10)
    summarize_results(results, ll_records)
