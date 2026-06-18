"""
Plan-D scaffolding: turnkey harness for the controlled saturated-regime field study
that the paper names as its priority next step. This does NOT fabricate field data.
It provides:
  (1) a power analysis -> how many reviewer-cases are needed to detect a target
      SEDI<->depth correlation at given power;
  (2) a randomised load-injection design (the experimental schedule);
  (3) the depth-coding rubric as a machine-checkable spec;
  (4) the analysis function that, GIVEN collected (observables, coded-depth) rows,
      computes hybrid-baseline SEDI, the correlation, and the detection metrics.
Run as-is to print the design + power table; supply a CSV to run the analysis.
"""
import sys, math, json, numpy as np
from scipy import stats

# ---- (1) power analysis: n to detect Pearson r at two-sided alpha, given power ----
def n_for_correlation(r_target=0.5, alpha=0.05, power=0.80):
    z = stats.norm.ppf
    za = z(1 - alpha/2); zb = z(power)
    zr = 0.5*math.log((1+r_target)/(1-r_target))      # Fisher transform
    n = ((za + zb)/zr)**2 + 3
    return int(math.ceil(n))

def power_table():
    rows = {}
    for r in [0.3, 0.4, 0.5, 0.6]:
        rows[f"r={r}"] = {f"power={p}": n_for_correlation(r, power=p) for p in [0.8, 0.9]}
    return rows

# ---- (2) randomised load-injection schedule ----
def injection_schedule(weeks=12, baseline_weeks=4, seed=0):
    """Stepped-wedge: baseline period, then randomised weeks at elevated injected load
    so each unit experiences both below- and above-rho_c regimes."""
    rng = np.random.default_rng(seed)
    load = np.ones(weeks)                              # x baseline arrival multiplier
    treat_weeks = np.arange(baseline_weeks, weeks)
    rng.shuffle(treat_weeks)
    ramp = np.linspace(1.3, 2.2, len(treat_weeks))    # push utilisation above rho_c
    for w, mult in zip(sorted(treat_weeks), ramp):
        load[w] = mult
    return [{"week": int(w), "arrival_multiplier": round(float(load[w]), 2),
             "expected_regime": "saturated" if load[w] > 1.4 else "normal"}
            for w in range(weeks)]

# ---- (3) depth-coding rubric (ground truth) ----
DEPTH_RUBRIC = {
    "scale": "0..1, mean of K binary checklist items independently coded per case",
    "items": ["claim_verified_against_source", "edge_cases_considered",
              "alternative_explanation_checked", "evidence_documented",
              "decision_justified_in_writing"],
    "coding": "two raters per case; report Cohen's kappa; depth = mean(items)",
    "blinding": "raters blind to load condition and to SEDI value"}

# ---- (4) analysis on collected rows ----
def hybrid_sedi(g_lambda, g_L, g_alpha, base_alpha, window=6):
    g_lambda=np.asarray(g_lambda); g_L=np.asarray(g_L); g_alpha=np.asarray(g_alpha)
    T=len(g_lambda); out=np.ones(T)
    for i in range(T):
        lo=max(0,i-window); b=slice(lo,i) if i>lo else slice(0,1)
        gl=g_lambda[i]/np.mean(g_lambda[b]); gL=g_L[i]/np.mean(g_L[b])
        decoup=max(1.0, gl/gL); qual=1.0+max(0.0, g_alpha[i]/base_alpha-1.0)
        out[i]=1.0/(decoup*qual)
    return out

def analyse_csv(path):
    import csv
    rows=list(csv.DictReader(open(path)))
    gl=[float(r["g_lambda"]) for r in rows]; gL=[float(r["g_L"]) for r in rows]
    ga=[float(r["g_alpha"]) for r in rows]; depth=[float(r["coded_depth"]) for r in rows]
    base_alpha=float(np.mean(ga[:4]))
    s=hybrid_sedi(gl,gL,ga,base_alpha)
    r=stats.pearsonr(s,depth)[0]; rho=stats.spearmanr(s,depth)[0]
    sat=(np.array(depth)<0.5).astype(int)
    return dict(n=len(rows), pearson=round(float(r),3), spearman=round(float(rho),3),
                saturated_fraction=round(float(sat.mean()),3))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(analyse_csv(sys.argv[1]), indent=2))
    else:
        out = dict(power_table=power_table(),
                   injection_schedule=injection_schedule(),
                   depth_rubric=DEPTH_RUBRIC)
        print(json.dumps(out, indent=2))
