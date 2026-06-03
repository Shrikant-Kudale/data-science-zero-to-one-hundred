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


# ---- 1. Build a low-rank matrix + noise ------------------------------------------
m, n, r = 200, 300, 5
A = rng.normal(size=(m, r)) @ rng.normal(size=(r, n)) + 0.1 * rng.normal(size=(m, n))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("top 8 singular values:", s[:8].round(2))

# ---- 2. Eckart-Young check ----------------------------------------------------------
print(f"{'k':>3} {'||A-Ak||_F':>12} {'sqrt(sum s^2)':>14} {'spectral':>10} {'s_{k+1}':>9} {'compression':>12}")
for k in (1, 3, 5, 10, 50):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    fro = np.linalg.norm(A - Ak); theo = np.sqrt((s[k:] ** 2).sum())
    assert np.isclose(fro, theo)
    print(f"{k:>3} {fro:12.4f} {theo:14.4f} {np.linalg.norm(A-Ak,2):10.4f} {s[k]:9.4f} {k*(m+n+1)/(m*n):12.2%}")

# ---- 3. Least squares: normal eq vs QR vs SVD on ill-conditioned X -------------
N, p = 1000, 6
base = rng.normal(size=(N, p)); X = base.copy()
X[:, 5] = X[:, 4] + 1e-7 * rng.normal(size=N)         # near-collinear column
w_true = np.array([1., -2., 0.5, 3., 1., 1.])
y = X @ w_true + 1e-3 * rng.normal(size=N)
print(f"cond(X)={np.linalg.cond(X):.2e}  cond(X'X)={np.linalg.cond(X.T@X):.2e}")
w_ne = np.linalg.solve(X.T @ X, X.T @ y)
Q, R = np.linalg.qr(X); w_qr = np.linalg.solve(R, Q.T @ y)
Ux, sx, Vtx = np.linalg.svd(X, full_matrices=False)
w_svd = Vtx.T @ ((Ux.T @ y) / sx)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
for name, w in [("normal eq", w_ne), ("QR", w_qr), ("SVD", w_svd), ("lstsq", w_lstsq)]:
    print(f"{name:10s} resid={np.linalg.norm(X@w-y):.5f}  |w|={np.linalg.norm(w):10.3f}  w[4:]={w[4:].round(3)}")
# truncated SVD = implicit regularization
keep = sx > 1e-4 * sx[0]
w_trunc = Vtx.T[:, keep] @ ((Ux.T @ y)[keep] / sx[keep])
print("truncated-SVD w:", w_trunc.round(3), "(stable, splits the collinear effect)")

# ---- 4. Cholesky sampling of correlated Gaussians ------------------------------
Sigma = np.array([[1, .8], [.8, 1.]]); L = np.linalg.cholesky(Sigma)
S = (L @ rng.normal(size=(2, 100000))).T
print("sample corr via Cholesky:", np.corrcoef(S.T)[0, 1].round(3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].semilogy(s, "o-", ms=3); ax[0].set_title("Singular value spectrum"); ax[0].set_xlabel("index")
ax[1].plot(np.cumsum(s**2) / np.sum(s**2)); ax[1].axhline(0.99, ls="--", c="gray"); ax[1].set_title("Cumulative energy")
savefig("svd_spectrum.png")
