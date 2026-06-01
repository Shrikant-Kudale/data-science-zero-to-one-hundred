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


def timeit(fn, *args, n=3):
    best, res = 1e9, None
    for _ in range(n):
        t = time.perf_counter(); res = fn(*args); best = min(best, time.perf_counter() - t)
    return best, res

# ---- 1. Python loop vs vectorized ------------------------------------------------
x = rng.normal(size=1_000_000)
def loop_sumsq(v):
    s = 0.0
    for i in v:
        s += i * i
    return s
t_loop, r1 = timeit(loop_sumsq, x, n=1)
t_temp, r2 = timeit(lambda v: (v ** 2).sum(), x)
t_dot, r3 = timeit(lambda v: np.dot(v, v), x)
print(f"loop={t_loop:.4f}s  square+sum={t_temp:.4f}s  dot={t_dot:.5f}s  speedup(dot)={t_loop/t_dot:,.0f}x")
assert np.allclose([r1, r2], r3)

# ---- 2. Memory layout & strides --------------------------------------------------
A = rng.normal(size=(3000, 3000))
print("C-order strides:", A.strides, "| F-order strides:", np.asfortranarray(A).strides)
print("A.T is a view sharing memory:", np.shares_memory(A, A.T), "| flags C/F:", A.T.flags.c_contiguous, A.T.flags.f_contiguous)
t_row, _ = timeit(lambda: A.sum(axis=1))
t_col, _ = timeit(lambda: A.sum(axis=0))
t_fcol, _ = timeit(lambda M: M.sum(axis=0), np.asfortranarray(A))
print(f"C-array sum axis=1 (contiguous): {t_row*1e3:.1f} ms | axis=0: {t_col*1e3:.1f} ms | F-array axis=0: {t_fcol*1e3:.1f} ms")

# ---- 3. Views vs copies ----------------------------------------------------------
v = A[::2, ::2]; c = A[[0, 2, 4]]; m = A[A > 3]
print("slice view shares memory:", np.shares_memory(A, v), "| fancy index copy shares:", np.shares_memory(A, c),
      "| boolean mask shares:", np.shares_memory(A, m))
B = np.arange(6); view = B[1:4]; view[0] = 99
print("mutating a view changed the base array:", B)

# ---- 4. dtype memory -------------------------------------------------------------
n = 1_000_000
for dt in (np.float64, np.float32, np.float16, np.int8):
    print(f"{np.dtype(dt).name:8s} {np.zeros(n, dtype=dt).nbytes/1e6:6.1f} MB")
big32 = np.full(10_000_000, 0.1, dtype=np.float32)
print("float32 sum:", big32.sum(), "| float64-accumulated:", big32.sum(dtype=np.float64), "| truth: 1000000.0")

# ---- 5. Rolling mean three ways --------------------------------------------------
s = rng.normal(size=200_000); w = 50
t_sw, a = timeit(lambda: np.lib.stride_tricks.sliding_window_view(s, w).mean(-1))
def cumsum_roll():
    c = np.cumsum(np.insert(s, 0, 0.0)); return (c[w:] - c[:-w]) / w
t_cs, b = timeit(cumsum_roll)
print(f"sliding_window_view O(nw): {t_sw*1e3:.1f} ms | cumsum O(n): {t_cs*1e3:.1f} ms | allclose={np.allclose(a, b)}")

# ---- plot -----------------------------------------------------------------------
labels = ["py loop", "x**2 .sum", "np.dot"]
plt.figure(figsize=(6, 3.5)); plt.bar(labels, [t_loop, t_temp, t_dot], color=["#c0392b", "#2980b9", "#27ae60"])
plt.yscale("log"); plt.ylabel("seconds (log)"); plt.title("Sum of squares, 1M elements"); savefig("vectorization_speed.png")
