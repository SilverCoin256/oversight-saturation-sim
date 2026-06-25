import os as _os; _os.makedirs("figures", exist_ok=True)
"""Regenerate the two core figures (saturation transitions; SEDI noise-stress) as REAL
plots, so the revised manuscript is self-contained. Uses the same model as the paper."""
import math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from experiments import build_scaling_trace, sedi, degradation_exp, kingman_lq

FIG = "figures"
rng = np.random.default_rng(7)

# ---- Fig: saturation transitions (Kingman bound vs event-driven M/G/1) ----
rho = np.linspace(0.1, 0.95, 14); cs2 = 1.69; Phi = (1+cs2)/2
Lq_king = kingman_lq(rho, 1.0, cs2*np.ones_like(rho))
# light event-driven M/G/1 per rho
def mg1_mean_wait(rho, cs2, n=200000, warm=20000, seed=1):
    rng = np.random.default_rng(seed); sigma = math.sqrt(math.log(1+cs2)); mu=1.0
    lam = rho*mu; t=0.0; na=rng.exponential(1/lam); q=0; busy=False; nd=math.inf
    area=0.0; last=0.0
    def svc():
        m=1/mu; return rng.lognormal(math.log(m)-sigma**2/2, sigma)
    ev=0
    while ev<n:
        if na<=nd: tn=na; typ='a'
        else: tn=nd; typ='d'
        if tn>warm:
            area += max(0,q-(1 if busy else 0))*(tn-max(last,warm))
        last=tn; t=tn; ev+=1
        if typ=='a':
            q+=1; na=t+rng.exponential(1/lam)
            if not busy: busy=True; nd=t+svc()
        else:
            q-=1; busy=False; nd=math.inf
            if q>0: busy=True; nd=t+svc()
    return area/(n_time(n,warm))
def n_time(n,warm):  # crude: not used; we time by events->approx via last
    return 1.0
# simpler: time-based sim
def mg1(rhov, cs2, T=300000, warm=30000, seed=1):
    rng=np.random.default_rng(seed); sigma=math.sqrt(math.log(1+cs2)); mu=1.0; lam=rhov*mu
    t=0.0; na=rng.exponential(1/lam); q=0; busy=False; nd=math.inf; area=0.0; last=0.0
    while t<T:
        if na<=nd: tn=na; typ='a'
        else: tn=nd; typ='d'
        if tn>warm: area+=max(0,q-(1 if busy else 0))*(tn-max(last,warm))
        last=tn; t=tn
        if typ=='a':
            q+=1; na=t+rng.exponential(1/lam)
            if not busy: busy=True; nd=t+rng.lognormal(math.log(1/mu)-sigma**2/2,sigma)
        else:
            q-=1; busy=False; nd=math.inf
            if q>0: busy=True; nd=t+rng.lognormal(math.log(1/mu)-sigma**2/2,sigma)
    return area/(T-warm)
Lq_sim = np.array([mg1(r, cs2, seed=int(1000*r)+1) for r in rho])
D = degradation_exp(rho)
fig, ax = plt.subplots(1,2,figsize=(9,3.5))
ax[0].plot(rho, Lq_king, color="#0077bb", label="Kingman G/G/1 bound")
ax[0].scatter(rho, Lq_sim, s=18, color="#cc3311", label="event-driven M/G/1")
ax[0].axvline(0.60, color="k", ls=":", lw=0.8); ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"mean $L_q$")
ax[0].set_title("Queue acceleration"); ax[0].legend(fontsize=8)
ax[1].plot(rho, D, color="#009988"); ax[1].axvline(0.60,color="k",ls=":",lw=0.8)
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel("review depth $D(\\rho)$"); ax[1].set_title("Depth collapse")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_saturation.png", dpi=300); plt.close(fig)
king_err = float(np.max(np.abs(Lq_king[:-1]-Lq_sim[:-1])/Lq_sim[:-1]))
print("max |Kingman-sim|/sim for rho<=0.9:", round(king_err,3))

# ---- Fig: SEDI noise-stress (non-circular E2-style on the paper DGP) ----
levels = [5,10,20,30,40]; pear=[]; spear=[]; rmse=[]
for s in levels:
    tr = build_scaling_trace(seed=7, noise=s/100.0)
    sv = sedi(tr["g_lambda"], tr["g_L"], tr["g_alpha"]); D=tr["D"]
    pear.append(stats.pearsonr(sv,D)[0]); spear.append(stats.spearmanr(sv,D)[0])
    rmse.append(float(np.sqrt(np.mean((sv-D)**2))))
fig, ax = plt.subplots(figsize=(5,3.6)); ax2=ax.twinx()
ax.plot(levels, pear, "o-", color="#0077bb", label="Pearson r")
ax.plot(levels, spear, "s--", color="#33bbee", label="Spearman")
ax2.plot(levels, rmse, "^:", color="#ee7733", label="RMSE")
ax.axhline(0.70, color="k", ls=":", lw=0.8)
ax.set_xlabel("noise level (%)"); ax.set_ylabel("correlation"); ax2.set_ylabel("RMSE")
ax.set_ylim(0,1); ax.legend(loc="lower left", fontsize=8); ax.set_title("SEDI noise-stress")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_noise.png", dpi=300); plt.close(fig)
import json
print(json.dumps(dict(kingman_max_relerr=round(king_err,3),
                      noise_pearson=dict(zip(levels,[round(p,3) for p in pear])),
                      noise_rmse=dict(zip(levels,[round(r,3) for r in rmse]))), indent=1))
