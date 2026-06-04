import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

def savefig(name):
    path = os.path.join(OUT, name)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f"[saved] {path}")


from sklearn.decomposition import PCA

# ---- 1. Correlated 5-D data -------------------------------------------------------
n, d = 1500, 5
mix = rng.normal(size=(d, d)); X = rng.normal(size=(n, d)) @ mix
Xc = X - X.mean(0)

# ---- 2. Eigen-decomposition of covariance -------------------------------------------
S = Xc.T @ Xc / (n - 1)
evals, evecs = np.linalg.eigh(S)
idx = np.argsort(evals)[::-1]; evals, evecs = evals[idx], evecs[:, idx]
evr = evals / evals.sum()
print("eigenvalues:", evals.round(3)); print("explained var ratio:", evr.round(3), "| cumulative:", np.cumsum(evr).round(3))

# ---- 3. SVD route gives the same answer -------------------------------------------
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("SVD eigenvalues match:", np.allclose(s ** 2 / (n - 1), evals))

# ---- 4. Compare to sklearn (up to sign) ------------------------------------------
pca = PCA(n_components=d).fit(X)
sign = np.sign((pca.components_ * evecs.T).sum(1))
print("sklearn components match:", np.allclose(pca.components_, (evecs.T * sign[:, None]), atol=1e-6),
      "| EVR match:", np.allclose(pca.explained_variance_ratio_, evr))

# ---- 5. Reconstruction error == discarded variance ------------------------------
k = 2
W = evecs[:, :k]; T = Xc @ W; Xhat = T @ W.T
mse_total = ((Xc - Xhat) ** 2).sum() / (n - 1)
print(f"recon error (sum sq / (n-1)) = {mse_total:.4f} | discarded eigenvalues = {evals[k:].sum():.4f}")

# ---- 6. Whitening --------------------------------------------------------------
Tw = (Xc @ evecs) / np.sqrt(evals)
print("whitened covariance ~ I:", np.allclose(np.cov(Tw.T), np.eye(d), atol=1e-8))

# ---- 7. Scaling matters ---------------------------------------------------------
X_units = X.copy(); X_units[:, 0] *= 1000
print("PC1 EVR unscaled:", PCA(1).fit(X_units).explained_variance_ratio_.round(3),
      "| after standardizing:", PCA(1).fit((X_units - X_units.mean(0)) / X_units.std(0)).explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(range(1, d + 1), evr); ax[0].plot(range(1, d + 1), np.cumsum(evr), "ro-"); ax[0].set_title("Scree + cumulative EVR")
ax[1].scatter(T[:, 0], T[:, 1], s=4, alpha=.5); ax[1].set_title("Scores on PC1/PC2"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
savefig("pca_scratch.png")
