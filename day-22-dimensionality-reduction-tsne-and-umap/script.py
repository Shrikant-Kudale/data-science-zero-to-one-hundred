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


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True); Xs = StandardScaler().fit_transform(X); print("digits:", X.shape)
# ---- 1. Distance concentration demo ---------------------------------------------------------------------------------------------
for d in (2, 10, 100, 1000):
    P = rng.random((500, d)); D = np.sqrt(((P[:1] - P[1:]) ** 2).sum(1)); print(f"d={d:5d} (max-min)/min distance = {(D.max()-D.min())/D.min():.2f}")

# ---- 2. Embeddings ----------------------------------------------------------------------------------------------------------------------
emb = {}
emb["PCA"] = PCA(2, random_state=0).fit_transform(Xs)
X50 = PCA(30, random_state=0).fit_transform(Xs)
for perp in (5, 30, 100):
    t = time.perf_counter(); emb[f"t-SNE p={perp}"] = TSNE(2, perplexity=perp, init="pca", random_state=0, max_iter=750).fit_transform(X50)
    print(f"t-SNE perplexity={perp} done in {time.perf_counter()-t:.1f}s")
try:
    import umap
    emb["UMAP nn=15"] = umap.UMAP(n_neighbors=15, min_dist=.1, random_state=0).fit_transform(X50)
    emb["UMAP nn=50"] = umap.UMAP(n_neighbors=50, min_dist=.1, random_state=0).fit_transform(X50)
except Exception as e:
    print("umap unavailable:", e)

# ---- 3. Quantitative evaluation ---------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
for k, E in emb.items():
    tw = trustworthiness(X50, E, n_neighbors=10); acc = cross_val_score(KNeighborsClassifier(10), E, y, cv=5).mean()
    print(f"{k:14s} trustworthiness@10={tw:.3f}  5-fold kNN acc in 2-D={acc:.3f}")

cols = 3; rows = int(np.ceil(len(emb) / cols)); fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 4 * rows)); axs = np.atleast_1d(axs).ravel()
for a, (k, E) in zip(axs, emb.items()): a.scatter(E[:, 0], E[:, 1], c=y, cmap="tab10", s=4); a.set_title(k); a.set_xticks([]); a.set_yticks([])
for a in axs[len(emb):]: a.axis("off")
savefig("embeddings.png")
