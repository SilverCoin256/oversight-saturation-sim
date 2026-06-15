# Oversight-Saturation Simulation

Reproducibility artefact for the paper:

> **Capacity Degradation in Distributed Human-in-the-Loop AI Pipelines: A
> Queueing-Network Analysis of Oversight Saturation**

A single, dependency-light Python script reproduces every quantitative result in
the paper.

## What it does

**Part A — verification (Figure 1).** An event-driven `M/G/1` simulation with
log-normal service reproduces the saturation curve. The mean queue length matches
the Pollaczek–Khinchine / Kingman expression

```
E[L_q] = rho^2 / (1 - rho) * Phi,    Phi = (1 + C_s^2) / 2
```

and the review-depth collapse follows `D(rho) = D_max * exp(-k (rho - rho_c)_+)`.

**Part B — validation (SEDI recovers latent depth).** A synthetic *scaling
episode* is generated from a generative model that instantiates the failure
mechanisms of Section 5: decision volume grows while human review capacity grows
sublinearly (Assumption 2), so utilisation rises and review depth collapses.
Three boundary observables — artefact write rate, end-to-end latency, and dispute
rate — are emitted, **each with its own independent multiplicative measurement
noise**. The State-Estimation Degradation Index (SEDI, Eq. 15) is then computed
from the boundary observables *only* and compared against the **latent** review
depth, which the auditor never sees.

> The trace is synthetic, not field data. It tests whether the observable index
> recovers the latent variable it is meant to track through independent noise — a
> precondition for, not a substitute for, field validation.

## Headline numbers (seed 7)

| Quantity | Value |
|---|---|
| `Phi = (1 + C_s^2)/2` | `1.345` |
| Part A max. relative error vs P–K (`rho <= 0.9`) | `~11%` (single replication) |
| SEDI vs latent depth — Pearson `r` | `0.98` |
| SEDI vs latent depth — Spearman `rho` | `0.89` |
| OLS `SEDI ~ D` slope / intercept | `1.02` / `-0.06` |

## Install & run

```bash
pip install -r requirements.txt
python oversight_sim.py            # full run: stats + figures/ + results.json
python oversight_sim.py --seed 7   # set the master seed (default 7)
python oversight_sim.py --no-plot  # statistics only, no matplotlib
```

Outputs:

- `results.json` — all reported statistics and the down-sampled coordinates used
  for the paper figures.
- `figures/figure1_reproduction.png` — Part A.
- `figures/sedi_validation.png` — Part B (time series + scatter).

## Parameters

All model parameters are module-level constants in `oversight_sim.py` and match
Table 1 of the paper: `MU0=1.0`, `RHO_C=0.60`, `K=3.50`, `DMAX=1.0`,
`CS2=1.69`. The scaling-episode parameters (load growth, capacity exponent
`gamma`, dispute gain and lag, noise level) are arguments of `scaling_trace`.

## Citation

If you use this code, please cite the paper. A `CITATION` entry will be added on
publication.

## License

MIT — see [`LICENSE`](LICENSE).
