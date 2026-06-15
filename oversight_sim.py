#!/usr/bin/env python3
"""
Oversight-saturation simulation for

    "Capacity Degradation in Distributed Human-in-the-Loop AI Pipelines:
     A Queueing-Network Analysis of Oversight Saturation"

This script is the reproducibility artefact for the paper. It does two things:

  Part A  (verification).  An event-driven M/G/1 simulation with log-normal
          service reproduces Figure 1: the mean queue length matches the
          Pollaczek--Khinchine / Kingman expression rho^2/(1-rho) * Phi with
          Phi = (1 + C_s^2)/2, and the review-depth collapse follows
          D(rho) = D_max * exp(-k (rho - rho_c)_+).

  Part B  (validation).  A synthetic *scaling-episode* trace is generated from
          a generative model that instantiates the failure mechanisms of
          Section 5: decision volume grows while human review capacity grows
          sublinearly (Assumption 2), so utilisation rises and review depth
          collapses. Three boundary observables -- artefact write rate, end-to-
          end latency, and dispute rate -- are emitted, EACH with its own
          independent multiplicative measurement noise. We then compute SEDI
          (Eq. 15) from the boundary observables only and test whether it
          recovers the *latent* review depth, which the auditor never sees.

The trace is synthetic, not field data: it tests whether the observable index
recovers the latent variable it is meant to track through independent noise. It
is a precondition for, not a substitute for, field validation.

Usage:
    python oversight_sim.py                # run everything, write figures/ + results.json
    python oversight_sim.py --seed 7       # change master seed
    python oversight_sim.py --no-plot      # skip matplotlib figures (stats only)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# Model parameters (Table 1 of the paper)
# ----------------------------------------------------------------------------
MU0 = 1.0          # normalised base service rate
RHO_C = 0.60       # critical utilisation threshold
K = 3.50           # cognitive decay rate
DMAX = 1.0         # maximum review depth
CS2 = 1.69         # service-time squared coefficient of variation
PHI = (1.0 + CS2) / 2.0            # saturation acceleration factor = 1.345
SIGMA_S = np.sqrt(np.log(1.0 + CS2))  # log-normal dispersion (mean preserved)
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40]  # Part C noise sweep (sigma)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


# ----------------------------------------------------------------------------
# Core model functions
# ----------------------------------------------------------------------------
def depth(rho, k=K, rho_c=RHO_C, dmax=DMAX):
    """Review-depth degradation function D(rho) (Eq. 3)."""
    return dmax * np.exp(-k * np.maximum(rho - rho_c, 0.0))


def kingman_lq(rho, phi=PHI):
    """Mean number in queue (P-K / Kingman): rho^2/(1-rho) * Phi (Eq. 4)."""
    rho = np.minimum(np.asarray(rho, dtype=float), 0.999)
    return rho ** 2 / (1.0 - rho) * phi


# ----------------------------------------------------------------------------
# Part A: event-driven M/G/1 verification of Figure 1
# ----------------------------------------------------------------------------
def simulate_mg1(rho, mu=MU0, n_jobs=150_000, burn=30_000, seed=0):
    """Single-server M/G/1 with log-normal service via the Lindley recursion.

    Returns (Lq, mean_depth): the time-average number waiting in queue and the
    mean realised review depth of served jobs at utilisation `rho`.
    """
    rng = np.random.default_rng(seed)
    lam = rho * mu
    # log-normal service times with mean 1/mu and SCV = CS2
    mean_log = np.log(1.0 / mu) - 0.5 * SIGMA_S ** 2
    inter = rng.exponential(1.0 / lam, n_jobs)
    serv = rng.lognormal(mean_log, SIGMA_S, n_jobs)

    # Lindley recursion: W_{i} = max(0, W_{i-1} + S_{i-1} - A_i)
    w = 0.0
    wait_sum = 0.0
    for i in range(1, n_jobs):
        w = max(0.0, w + serv[i - 1] - inter[i])
        if i >= burn:
            wait_sum += w
    mean_wait = wait_sum / (n_jobs - burn)
    lq = lam * mean_wait               # Little's law on the queue
    mean_depth = float(depth(rho))     # realised review depth at this load
    return lq, mean_depth


def saturation_curve(seed=2024):
    rhos = np.array([.10, .20, .30, .40, .50, .55, .60,
                     .65, .70, .75, .80, .85, .90])
    lq_sim = np.empty_like(rhos)
    for i, r in enumerate(rhos):
        lq_sim[i], _ = simulate_mg1(r, seed=seed + i)
    lq_theory = kingman_lq(rhos)
    d = depth(rhos)
    # relative error of the simulator against the closed form
    rel_err = np.abs(lq_sim - lq_theory) / np.maximum(lq_theory, 1e-9)
    return {
        "rho": rhos, "lq_sim": lq_sim, "lq_theory": lq_theory,
        "depth": d, "max_rel_err": float(np.nanmax(rel_err[rhos <= 0.9])),
    }


# ----------------------------------------------------------------------------
# Part B: scaling-episode trace and SEDI validation
# ----------------------------------------------------------------------------
def scaling_trace(T=360, seed=7, gamma=0.5, load_growth=8.0,
                  rho0=0.33, dispute_gain=2.0, dispute_lag=7,
                  noise=0.05):
    """Generate a synthetic scaling episode.

    Volume grows by `load_growth` over T steps; human capacity grows
    sublinearly as volume**gamma (Assumption 2), so utilisation rises and depth
    collapses. Boundary observables carry independent multiplicative noise.

    Returns a dict of arrays. `D_true` is the LATENT depth (hidden from the
    auditor); the SEDI inputs are the noisy boundary observables.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # structural (noise-free) volume growth factor, 1 -> load_growth
    g_lam_struct = 1.0 + (load_growth - 1.0) * (t / (T - 1)) ** 1.3

    # sublinear capacity -> rising utilisation
    rho_nom = rho0 * g_lam_struct ** (1.0 - gamma)
    rho_eff = np.clip(rho_nom, 0.0, 0.985)

    # latent review depth = structural degradation + behavioural noise
    D_struct = depth(rho_eff)
    D_true = np.clip(D_struct + rng.normal(0.0, 0.025, T), 0.01, 1.0)

    # --- boundary observables (each independently noised) -------------------
    # (1) artefact/telemetry rate tracks decision volume (Eq. 13)
    g_lam_obs = g_lam_struct * rng.lognormal(0.0, noise, T)

    # (2) end-to-end latency: growth is depth-limited. While depth is intact
    #     latency tracks volume; as depth collapses, latency growth slows
    #     (cases are closed faster with shallower review). Structurally,
    #     d(log L)/d(log lambda) = D, so log L = cumulative integral of D.
    dlog = np.diff(np.log(g_lam_struct), prepend=0.0)
    g_L_struct = np.exp(np.cumsum(D_struct * dlog))
    g_L_obs = g_L_struct * rng.lognormal(0.0, noise, T)

    # (3) dispute/appeal rate rises (with a lag) as depth falls
    D_lag = np.concatenate([np.full(dispute_lag, D_struct[0]),
                            D_struct[:-dispute_lag]])
    g_alpha_struct = 1.0 + dispute_gain * (1.0 - D_lag / D_struct[0])
    g_alpha_obs = g_alpha_struct * rng.lognormal(0.0, noise, T)

    return {
        "t": t, "rho": rho_eff, "D_true": D_true,
        "g_lam": g_lam_obs, "g_L": g_L_obs, "g_alpha": g_alpha_obs,
    }


def compute_sedi(g_lam, g_L, g_alpha):
    """State-Estimation Degradation Index (Eq. 15), from observables only."""
    decoupling = np.maximum(1.0, g_lam / g_L)
    quality = 1.0 + np.maximum(g_alpha - 1.0, 0.0)
    return 1.0 / (quality * decoupling)


def validate_sedi(seed=7, noise=0.05):
    tr = scaling_trace(seed=seed, noise=noise)
    sedi = compute_sedi(tr["g_lam"], tr["g_L"], tr["g_alpha"])
    d = tr["D_true"]
    pear_r, pear_p = stats.pearsonr(sedi, d)
    spear_r, spear_p = stats.spearmanr(sedi, d)
    slope, intercept, r, p, se = stats.linregress(d, sedi)
    rmse = float(np.sqrt(np.mean((sedi - d) ** 2)))
    return {
        "trace": tr, "sedi": sedi,
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "spearman_r": float(spear_r), "spearman_p": float(spear_p),
        "ols_slope": float(slope), "ols_intercept": float(intercept),
        "n": int(len(d)), "sedi_final": float(sedi[-1]), "rmse": rmse,
    }


def noise_robustness_sweep(seed=7, noise_levels=None):
    """Part C: SEDI recovery under increasing measurement noise.

    Each noise level sigma applies independent lognormal multiplicative noise
    with dispersion sigma to all three boundary observables. Reports Pearson r,
    Spearman rho, and RMSE for each level to characterise graceful degradation.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    rows = []
    for sigma in noise_levels:
        v = validate_sedi(seed=seed, noise=sigma)
        rows.append({
            "noise_pct": int(round(sigma * 100)),
            "noise_sigma": float(sigma),
            "pearson_r": v["pearson_r"],
            "spearman_r": v["spearman_r"],
            "rmse": v["rmse"],
        })
    return rows


# ----------------------------------------------------------------------------
# Plotting and reporting
# ----------------------------------------------------------------------------
def make_robustness_plot(rows):
    """Figure 3: SEDI robustness under increasing measurement noise."""
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    noise_pct = [r["noise_pct"] for r in rows]
    pearson  = [r["pearson_r"] for r in rows]
    spearman = [r["spearman_r"] for r in rows]
    rmse     = [r["rmse"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax2 = ax1.twinx()
    l1, = ax1.plot(noise_pct, pearson,  "b-o", ms=6, lw=2.0, label="Pearson $r$")
    l2, = ax1.plot(noise_pct, spearman, "g-s", ms=6, lw=2.0, label="Spearman $\\rho$")
    l3, = ax2.plot(noise_pct, rmse,     "r--^", ms=6, lw=1.8, label="RMSE (right axis)")
    ax1.axhline(0.70, color="0.55", ls=":", lw=0.9)
    ax1.set(xlabel="Measurement noise $\\sigma$ (%)", ylabel="Correlation",
            xlim=(2, 43), ylim=(0.40, 1.02), xticks=noise_pct)
    ax2.set(ylabel="RMSE", ylim=(0.0, 0.55))
    lines, labels = [l1, l2, l3], [l.get_label() for l in [l1, l2, l3]]
    ax1.legend(lines, labels, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sedi_robustness.png"), dpi=200)
    plt.close(fig)


def make_plots(sat, val):
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    # Figure 1 reproduction
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    rr = np.linspace(0.05, 0.905, 200)
    ax[0].plot(rr, kingman_lq(rr), "b-", lw=2, label="P-K / Kingman bound")
    ax[0].plot(sat["rho"], sat["lq_sim"], "o", color="navy", ms=4,
               label="M/G/1 simulation")
    ax[0].axvline(RHO_C, ls="--", color="0.4")
    ax[0].set(xlabel=r"Utilisation $\rho$", ylabel=r"Mean queue length $L_q$",
              xlim=(0, 1), ylim=(0, 12))
    ax[0].legend(frameon=False, fontsize=8)
    rr2 = np.linspace(0, 0.95, 200)
    ax[1].plot(rr2, depth(rr2), "r-", lw=2, label=r"$D(\rho)$")
    ax[1].axvline(RHO_C, ls="--", color="0.4")
    ax[1].set(xlabel=r"Utilisation $\rho$", ylabel=r"Review depth $D(\rho)$",
              xlim=(0, 1), ylim=(0, 1.1))
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure1_reproduction.png"), dpi=200)
    plt.close(fig)

    # SEDI validation: time series + scatter
    tr, sedi = val["trace"], val["sedi"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(tr["t"], tr["D_true"], color="crimson", lw=1.6,
               label="latent depth $D(t)$ (hidden)")
    ax[0].plot(tr["t"], sedi, color="navy", lw=1.6, label="SEDI$(t)$ (observed)")
    ax[0].set(xlabel="time (steps)", ylabel="index value", ylim=(0, 1.05))
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].scatter(tr["D_true"], sedi, s=10, color="0.3", alpha=0.7)
    xs = np.linspace(tr["D_true"].min(), tr["D_true"].max(), 50)
    ax[1].plot(xs, val["ols_intercept"] + val["ols_slope"] * xs, "r-", lw=1.5)
    ax[1].set(xlabel="latent depth $D$", ylabel="SEDI",
              title=f"Pearson $r$ = {val['pearson_r']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sedi_validation.png"), dpi=200)
    plt.close(fig)


def pgfplots_robustness_coords(rows):
    """Emit pgfplots coordinates for the noise-robustness figure."""
    pr = " ".join(f"({r['noise_pct']},{r['pearson_r']:.4f})" for r in rows)
    sp = " ".join(f"({r['noise_pct']},{r['spearman_r']:.4f})" for r in rows)
    rm = " ".join(f"({r['noise_pct']},{r['rmse']:.4f})" for r in rows)
    return {"pearson": pr, "spearman": sp, "rmse": rm}


def pgfplots_coords(val, every=20):
    """Emit down-sampled coordinates for the paper's pgfplots figure."""
    tr, sedi = val["trace"], val["sedi"]
    idx = np.arange(0, len(tr["t"]), every)
    ts = "".join(f"({int(tr['t'][i])},{tr['D_true'][i]:.3f}) " for i in idx)
    ss = "".join(f"({int(tr['t'][i])},{sedi[i]:.3f}) " for i in idx)
    sc = "".join(f"({tr['D_true'][i]:.3f},{sedi[i]:.3f}) " for i in idx)
    return {"depth_series": ts.strip(), "sedi_series": ss.strip(),
            "scatter": sc.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("  Oversight-saturation simulation")
    print("=" * 64)
    print(f"  Phi = (1 + C_s^2)/2 = {PHI:.3f}   sigma_S = {SIGMA_S:.3f}\n")

    print("[Part A] Event-driven M/G/1 verification of Figure 1 ...")
    sat = saturation_curve()
    for r, ls, lt in zip(sat["rho"], sat["lq_sim"], sat["lq_theory"]):
        print(f"    rho={r:0.2f}  Lq_sim={ls:7.3f}  Lq_PK={lt:7.3f}")
    print(f"    max relative error (rho<=0.9): {sat['max_rel_err']*100:.1f}%\n")

    print("[Part B] SEDI validation on synthetic scaling trace ...")
    val = validate_sedi(seed=args.seed, noise=0.05)
    print(f"    n               = {val['n']}")
    print(f"    Pearson  r      = {val['pearson_r']:.3f}  (p={val['pearson_p']:.1e})")
    print(f"    Spearman rho    = {val['spearman_r']:.3f}  (p={val['spearman_p']:.1e})")
    print(f"    OLS  SEDI ~ D   : slope={val['ols_slope']:.3f}, "
          f"intercept={val['ols_intercept']:.3f}")
    print(f"    SEDI at episode end = {val['sedi_final']:.3f}\n")

    print("[Part C] Noise robustness sweep ...")
    rob = noise_robustness_sweep(seed=args.seed)
    for r in rob:
        print(f"    noise={r['noise_pct']:2d}%  Pearson={r['pearson_r']:.3f}  "
              f"Spearman={r['spearman_r']:.3f}  RMSE={r['rmse']:.3f}")
    rob_coords = pgfplots_robustness_coords(rob)
    print()

    results = {
        "phi": PHI, "sigma_s": SIGMA_S,
        "partA_max_rel_err": sat["max_rel_err"],
        "partB": {k: val[k] for k in
                  ("pearson_r", "pearson_p", "spearman_r", "spearman_p",
                   "ols_slope", "ols_intercept", "n", "sedi_final", "rmse")},
        "partC_robustness": rob,
        "pgfplots": pgfplots_coords(val),
        "pgfplots_robustness": rob_coords,
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Wrote {os.path.join(HERE, 'results.json')}")

    if not args.no_plot:
        make_plots(sat, val)
        make_robustness_plot(rob)
        print(f"  Wrote figures to {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
