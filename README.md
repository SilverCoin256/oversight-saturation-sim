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

**Part C — noise robustness (graceful degradation).** The full scaling episode is
re-run at five measurement-noise levels (`sigma = 5, 10, 20, 30, 40%`) applied
independently to all three boundary observables. Pearson `r`, Spearman `rho`,
and RMSE are reported per level so the headline correlation is presented as a
point on a characterised curve rather than a single number.

**Real-data study (`realdata_github.py`) — computability & specificity.** SEDI is
applied to real operational telemetry from a large public code-review pipeline
(`scikit-learn`, 3,000 PRs, 22 months). The three boundary observables
(submission rate, median hours-to-merge, closed-unmerged abandonment) are
computed per month, while an **independent** review-depth proxy (formal reviews +
inline review threads per merged PR) is held out and never fed to SEDI. The study
shows (i) SEDI is computable end-to-end from real public telemetry with no latent
variable; (ii) the pipeline operates **below saturation** (load vs depth
uncorrelated, `r=0.13`, n.s.); and (iii) a fixed baseline **false-alarms**
(median 0.07, 95% of months) because the project's latency *fell* as volume
*rose* — a secular efficiency trend the fixed-baseline decoupling term misreads —
while a principled **rolling baseline** restores specificity (median 0.83). This
is a real-world *computability and specificity* result, **not** a positive
saturation-detection result: the public history contained no saturation episode.
A saturated-regime field study remains future work.

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

# Synthetic results (Parts A, B, C) — no network needed
python oversight_sim.py            # full run: stats + figures/ + results.json
python oversight_sim.py --seed 7   # set the master seed (default 7)
python oversight_sim.py --no-plot  # statistics only, no matplotlib

# Real-data study — needs a read-only GitHub token (public_repo / read scope)
export GITHUB_TOKEN=ghp_xxx
python realdata_github.py --repo scikit-learn/scikit-learn --max-prs 3000 --min-month 10
```

Outputs:

- `results.json` — synthetic statistics (Parts A/B/C) and figure coordinates.
- `realdata_results.json` — real-data specificity statistics and coordinates.
- `figures/figure1_reproduction.png` — Part A.
- `figures/sedi_validation.png` — Part B (time series + scatter).
- `figures/sedi_robustness.png` — Part C (noise sweep).
- `figures/sedi_realdata.png` — real-data computability/specificity.

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

---

## Revision experiments (added 2026-06)

Five additional scripts extend the original artefact with the experiments requested in
peer review. They are dependency-light (same `requirements.txt`) and seed-fixed.

| Script | Produces | Headline result |
|---|---|---|
| `experiments.py` | E1–E6 → `results.json`, `figures/` | QNA vs discrete-event sim **0.3%** error; non-circular SEDI recovery **r=0.82**; **hybrid-baseline detector recall 0.88 / FPR 0.00 / AUC 1.00**; capacity-gate fix verified |
| `figs_core.py` | saturation + noise-stress figures | Kingman vs M/G/1 agreement; graceful noise degradation |
| `realdata_multi.py` | multi-repo SEDI (`realdata_multi_results.json`) | SEDI computable across several real review pipelines (cached GitHub API) |
| `search_timeseries.py` | multi-year monthly SEDI (`timeseries_*.json`) | **Real load-surge detection**: `langchain` 33× surge → hybrid SEDI 0.015 at the 2023-07 onset |
| `field_harness.py` | controlled-study design | Power analysis (30 reviewer-cases/regime for r=0.5@80%), injection schedule, depth-coding rubric |

Run:
```bash
pip install -r requirements.txt
python3 experiments.py            # offline, ~30s
python3 figs_core.py              # offline
python3 search_timeseries.py langchain-ai/langchain 2022-10 2024-06   # GitHub search API
python3 field_harness.py          # study design + power table
```

### Key methodological addition: hybrid baselining
The latency/throughput "decoupling" factor is confounded—process efficiency and oversight
degradation share a boundary signature. A rolling baseline removes the efficiency false
alarm but also suppresses detection. **Hybrid baselining** (rolling baseline for the
efficiency-confounded decoupling factor, fixed-absolute baseline for the harm-signalling
dispute rate) restores detection (AUC 0.62→1.00) while preserving specificity. See
`experiments.py::E4b_gated_detector`.

### Integrity note
Synthetic and agent-based traces are labelled as such. The real-data studies use a measured
depth *proxy* (review-thread comments per merged PR), not human-coded ground-truth depth;
the controlled study in `field_harness.py` is the design for collecting the latter.
