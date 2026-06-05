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


from scipy import signal, optimize, fft

# ---- 1. Noisy two-tone signal ------------------------------------------------------
fs = 500; t = np.arange(0, 2, 1 / fs)
clean = np.sin(2 * np.pi * 5 * t)
noisy = clean + 0.6 * np.sin(2 * np.pi * 60 * t) + 0.3 * rng.normal(size=t.size)
sos = signal.butter(4, 15, btype="low", fs=fs, output="sos")
filt = signal.sosfiltfilt(sos, noisy)
print(f"RMSE noisy->clean = {np.sqrt(np.mean((noisy-clean)**2)):.3f} | filtered->clean = {np.sqrt(np.mean((filt-clean)**2)):.3f}")

# ---- 2. FFT spectrum + peak detection ------------------------------------------------
spec = np.abs(fft.rfft(noisy * np.hanning(t.size))) * 2 / t.size; freqs = fft.rfftfreq(t.size, 1 / fs)
pk, _ = signal.find_peaks(spec, height=0.15)
print("dominant frequencies (Hz):", freqs[pk].round(2), "| resolution df =", fs / t.size, "Hz")

# ---- 3. Logistic regression MLE with analytic gradient -----------------------------
N, p = 5000, 4
X = np.c_[np.ones(N), rng.normal(size=(N, p - 1))]; w_true = np.array([-0.5, 1.5, -2.0, 0.7])
y = rng.binomial(1, 1 / (1 + np.exp(-X @ w_true)))
def nll(w):
    z = X @ w; return np.sum(np.logaddexp(0, z) - y * z)
def grad(w): return X.T @ (1 / (1 + np.exp(-X @ w)) - y)
res = optimize.minimize(nll, np.zeros(p), jac=grad, method="BFGS")
p_hat = 1 / (1 + np.exp(-X @ res.x)); H = X.T @ (X * (p_hat * (1 - p_hat))[:, None]); se = np.sqrt(np.diag(np.linalg.inv(H)))
print("MLE w:", res.x.round(3), "| true:", w_true, "| SE:", se.round(3), "| iters:", res.nit)
print("grad check err:", optimize.check_grad(nll, grad, np.zeros(p)).round(4))

# ---- 4. curve_fit: exponential decay with uncertainty -------------------------------
tt = np.linspace(0, 5, 80); yy = 3.0 * np.exp(-1.2 * tt) + 0.5 + 0.05 * rng.normal(size=tt.size)
popt, pcov = optimize.curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, tt, yy, p0=[1, 1, 0])
print("curve_fit (a,b,c):", popt.round(3), "+/-", np.sqrt(np.diag(pcov)).round(3), "| truth (3, 1.2, 0.5)")

# ---- 5. Constrained min-variance portfolio (SLSQP) ----------------------------------
Sigma = np.cov(rng.normal(size=(250, 4)) @ rng.normal(size=(4, 4)), rowvar=False)
cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
r = optimize.minimize(lambda w: w @ Sigma @ w, np.ones(4) / 4, bounds=[(0, 1)] * 4, constraints=cons, method="SLSQP")
print("min-var weights:", r.x.round(3), "| variance:", round(r.fun, 4), "| equal-weight var:", round(np.ones(4) @ Sigma @ np.ones(4) / 16, 4))

# ---- 6. LP via HiGHS ---------------------------------------------------------------
lp = optimize.linprog(c=[-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], method="highs")
print("LP optimum x:", lp.x, "objective:", -lp.fun)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(t[:400], noisy[:400], alpha=.4, label="noisy"); ax[0].plot(t[:400], filt[:400], label="filtered"); ax[0].plot(t[:400], clean[:400], "k--", label="clean"); ax[0].legend(); ax[0].set_title("Butterworth zero-phase filter")
ax[1].plot(freqs, spec); ax[1].set_xlim(0, 100); ax[1].set_title("Spectrum"); ax[1].set_xlabel("Hz")
savefig("signal_filtering.png")
