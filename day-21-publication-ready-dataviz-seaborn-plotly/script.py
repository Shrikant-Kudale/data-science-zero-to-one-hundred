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


import pandas as pd
try:
    import seaborn as sns; HAVE_SNS = True
except ImportError:
    HAVE_SNS = False; print("seaborn missing -> matplotlib fallback")

# ---- 1. Data ---------------------------------------------------------------------------------------------------------------
n = 900
df = pd.DataFrame({"group": rng.choice(["Control", "Variant A", "Variant B"], n), "x": rng.normal(0, 1, n)})
df["spend"] = np.exp(3 + 0.25 * (df.group == "Variant A") + 0.45 * (df.group == "Variant B") + 0.35 * df.x + rng.normal(0, .5, n))
df["week"] = rng.integers(1, 9, n)
PAL = ["#7f8c8d", "#2980b9", "#e67e22"]                              # colorblind-friendly distinct hues
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold"})

# ---- 2. Multi-panel publication figure -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
order = ["Control", "Variant A", "Variant B"]
if HAVE_SNS:
    sns.violinplot(data=df, x="group", y="spend", order=order, palette=PAL, inner="quartile", cut=0, ax=ax[0, 0])
    sns.ecdfplot(data=df, x="spend", hue="group", hue_order=order, palette=PAL, ax=ax[0, 1])
    sns.regplot(data=df, x="x", y=np.log(df.spend), scatter_kws=dict(s=8, alpha=.4), line_kws=dict(color="k"), ax=ax[1, 0])
    sns.heatmap(df.pivot_table(index="group", columns="week", values="spend", aggfunc="median").loc[order], cmap="viridis", annot=True, fmt=".0f", cbar=False, ax=ax[1, 1])
else:
    ax[0, 0].boxplot([df.spend[df.group == g] for g in order], labels=order)
    for g, c in zip(order, PAL): ax[0, 1].plot(np.sort(df.spend[df.group == g]), np.linspace(0, 1, (df.group == g).sum()), c=c, label=g)
    ax[1, 0].scatter(df.x, np.log(df.spend), s=8, alpha=.4); ax[1, 1].imshow(df.pivot_table(index="group", columns="week", values="spend", aggfunc="median").loc[order], cmap="viridis")
ax[0, 0].set_title("Spend is right-skewed; Variant B shifts the whole distribution"); ax[0, 0].set_yscale("log")
ax[0, 1].set_title("ECDF: B stochastically dominates"); ax[1, 0].set_title("log(spend) vs engagement score"); ax[1, 1].set_title("Median spend by week")
fig.suptitle("Experiment readout: Variant B lifts spend", fontsize=14, y=1.0); savefig("publication_figure.png")

# ---- 3. Forest plot of lifts with bootstrap CIs ------------------------------------------------------------------------------------
ctrl = df.spend[df.group == "Control"].values; rows = []
for g in order[1:]:
    v = df.spend[df.group == g].values; bs = [v[rng.integers(0, len(v), len(v))].mean() / ctrl[rng.integers(0, len(ctrl), len(ctrl))].mean() - 1 for _ in range(2000)]
    rows.append((g, v.mean() / ctrl.mean() - 1, *np.percentile(bs, [2.5, 97.5])))
fr = pd.DataFrame(rows, columns=["g", "lift", "lo", "hi"]); print(fr.round(3))
plt.figure(figsize=(6, 2.8)); plt.errorbar(fr.lift, range(len(fr)), xerr=[fr.lift - fr.lo, fr.hi - fr.lift], fmt="o", c="#2980b9", capsize=4)
plt.axvline(0, c="k", ls="--"); plt.yticks(range(len(fr)), fr.g); plt.xlabel("relative lift in mean spend (95% CI)"); plt.title("Forest plot"); savefig("forest_plot.png")

# ---- 4. Vector export + interactive Plotly ---------------------------------------------------------------------------------------------
plt.figure(figsize=(4, 3)); plt.hist(np.log(df.spend), bins=30, color=PAL[1]); plt.savefig(os.path.join(OUT, "hist.svg")); plt.close(); print("[saved] hist.svg (vector)")
try:
    import plotly.express as px
    px.box(df, x="group", y="spend", color="group", log_y=True, points="outliers", title="Interactive spend by group").write_html(os.path.join(OUT, "interactive_box.html"))
    print("[saved] interactive_box.html")
except ImportError:
    print("plotly not installed -> skipping interactive output")
