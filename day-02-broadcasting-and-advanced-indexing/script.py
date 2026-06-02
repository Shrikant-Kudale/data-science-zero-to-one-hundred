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


n, m, d = 500, 10, 8
X = rng.normal(size=(n, d)); C = rng.normal(size=(m, d))

# ---- 1. Column standardization by broadcasting -------------------------------
Z = (X - X.mean(axis=0)) / X.std(axis=0)
print("standardized mean ~0:", np.abs(Z.mean(0)).max().round(12), "| std ~1:", Z.std(0).round(6))

# ---- 2. Pairwise distances: broadcast vs Gram trick ---------------------------
t = time.perf_counter(); D1 = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)); t1 = time.perf_counter() - t
t = time.perf_counter()
D2 = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T, 0)); t2 = time.perf_counter() - t
print(f"broadcast {t1*1e3:.2f} ms (temp tensor {n*m*d*8/1e6:.2f} MB) | gram {t2*1e3:.2f} ms | match={np.allclose(D1, D2)}")

# ---- 3. Top-k nearest centers per row with argpartition -----------------------
k = 3
idx = np.argpartition(D1, k, axis=1)[:, :k]
order = np.argsort(np.take_along_axis(D1, idx, axis=1), axis=1)
topk = np.take_along_axis(idx, order, axis=1)
assert (topk == np.argsort(D1, axis=1)[:, :k]).all()
print("top-3 centers for first 3 points:\n", topk[:3])

# ---- 4. Boolean masks, where, ix_ ---------------------------------------------
mask = (X[:, 0] > 0) & (X[:, 1] < 0)
print("rows matching mask:", mask.sum(), "| masked shape:", X[mask].shape)
Xc = np.where(np.abs(X) > 2.5, np.sign(X) * 2.5, X)   # winsorize via where
print("max abs after clip:", np.abs(Xc).max())
block = X[np.ix_([0, 1, 2], [0, 3])]
print("ix_ block shape:", block.shape, "| paired indexing shape:", X[[0, 1, 2], [0, 3, 4]].shape)

# ---- 5. Fancy-index gotcha: duplicate indices --------------------------------
a = np.zeros(4); a[[0, 0, 1]] += 1
b = np.zeros(4); np.add.at(b, [0, 0, 1], 1)
print("a[[0,0,1]] += 1 ->", a, "| np.add.at ->", b, "| bincount ->", np.bincount([0, 0, 1], minlength=4))

# ---- 6. Mixed slice/advanced index placement gotcha --------------------------
T = np.zeros((2, 3, 4))
print("T[:, [0,1], 0].shape =", T[:, [0, 1], 0].shape, "| T[[0,1], :, [0,1]].shape =", T[[0, 1], :, [0, 1]].shape)

plt.figure(figsize=(6, 4)); plt.imshow(D1[:60], aspect="auto", cmap="viridis"); plt.colorbar(label="distance")
plt.xlabel("center"); plt.ylabel("point"); plt.title("Pairwise distance matrix (first 60 points)"); savefig("distance_heatmap.png")
