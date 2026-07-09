#!/usr/bin/env python3
"""
Real-data validation of SEDI on an open-source code-review pipeline.

A GitHub pull-request (PR) review queue is a genuine *distributed human-in-the-
loop review pipeline*: automated contributors and CI emit artefacts (PRs, diffs,
checks) while a bounded set of human maintainers must review them. This script
tests whether SEDI -- computed only from BOUNDARY-OBSERVABLE telemetry -- tracks
an INDEPENDENTLY measured proxy for review depth on a large public repository.

Boundary observables (the only inputs SEDI is allowed to see), per month:
  * g_lambda : PR submission rate            (artefact write rate, Eq. 13)
  * g_L      : median hours-to-merge          (end-to-end latency)
  * g_alpha  : abandonment rate = closed-unmerged / total  (dispute/appeal proxy)

Independent review-depth proxy (NEVER given to SEDI; the "ground truth"):
  * D_proxy  : mean (formal reviews + inline review threads) per merged PR,
               normalised to its baseline. This counts human scrutiny per
               artefact and is not an input to the SEDI formula.

The substantive test: as submission load rises relative to review capacity,
does the measured per-PR scrutiny fall, and does observable-only SEDI track it?

This is real operational data from a real review pipeline. It is a real-world
*corroboration*; it is NOT a controlled study of a regulated-AI deployment,
which remains future work (see paper, Limitations).

Usage:
    export GITHUB_TOKEN=ghp_xxx          # needs only public_repo / read scope
    python realdata_github.py --repo microsoft/vscode --max-prs 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import requests
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
GQL = "https://api.github.com/graphql"

QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:50, states:[MERGED,CLOSED],
                 orderBy:{field:CREATED_AT, direction:DESC}, after:$cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        createdAt
        mergedAt
        state
        additions
        deletions
        reviews { totalCount }
        reviewThreads { totalCount }
      }
    }
  }
}
"""


def fetch_prs(owner, name, token, max_prs=2000):
    """Page through PRs via GraphQL; return a list of dicts."""
    headers = {"Authorization": f"bearer {token}"}
    cursor, out = None, []
    while len(out) < max_prs:
        data = None
        for attempt in range(6):
            try:
                r = requests.post(GQL, json={"query": QUERY,
                                  "variables": {"owner": owner, "name": name,
                                                "cursor": cursor}},
                                  headers=headers, timeout=60)
                if r.status_code in (502, 503, 504):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                if "errors" in data:
                    # transient GraphQL timeouts surface here too
                    time.sleep(2.0 * (attempt + 1))
                    data = None
                    continue
                break
            except requests.exceptions.RequestException:
                time.sleep(2.0 * (attempt + 1))
        if data is None:
            print(f"  stopping early after {len(out)} PRs (transient errors)")
            break
        conn = data["data"]["repository"]["pullRequests"]
        out.extend(conn["nodes"])
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.3)
    return out[:max_prs]


def monthly_series(prs, min_prs_per_month=15):
    """Aggregate PRs into a monthly time series of observables + depth proxy."""
    by_month = defaultdict(list)
    for p in prs:
        ts = p["createdAt"]
        month = ts[:7]  # YYYY-MM
        by_month[month].append(p)

    months = sorted(m for m, v in by_month.items()
                    if len(v) >= min_prs_per_month)
    rows = []
    for m in months:
        ps = by_month[m]
        n = len(ps)
        merged = [p for p in ps if p["state"] == "MERGED" and p["mergedAt"]]
        # latency: median hours-to-merge
        lat = []
        for p in merged:
            t0 = datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(p["mergedAt"].replace("Z", "+00:00"))
            lat.append((t1 - t0).total_seconds() / 3600.0)
        med_lat = float(np.median(lat)) if lat else np.nan
        # abandonment (dispute/appeal proxy): closed-unmerged fraction
        n_closed_unmerged = sum(1 for p in ps if p["state"] == "CLOSED")
        aband = n_closed_unmerged / n
        # depth proxy: mean (reviews + review threads) per merged PR
        depth = np.mean([p["reviews"]["totalCount"]
                         + p["reviewThreads"]["totalCount"] for p in merged]) \
            if merged else np.nan
        rows.append({"month": m, "n_prs": n, "submit": n,
                     "med_latency_h": med_lat, "abandon": aband,
                     "depth_proxy": float(depth)})
    return [r for r in rows if not (np.isnan(r["med_latency_h"])
                                    or np.isnan(r["depth_proxy"]))]


def _rolling_baseline(x, w=3):
    """Trailing-window median baseline (absorbs secular efficiency trends)."""
    b = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        b[i] = np.median(x[max(0, i - w):i]) if i > 0 else x[0]
    return b


def compute_sedi_real(rows, mode="fixed"):
    """SEDI from boundary observables only (Eq. 15).

    mode='fixed'   : baseline = first month (as defined in the paper).
    mode='rolling' : baseline = trailing 3-month median, so only ACUTE
                     decoupling trips SEDI and slow throughput-efficiency
                     trends are absorbed (the real-data refinement).
    """
    submit = np.array([r["submit"] for r in rows], dtype=float)
    lat = np.array([r["med_latency_h"] for r in rows], dtype=float)
    aband = np.array([r["abandon"] for r in rows], dtype=float)
    if mode == "rolling":
        b_s, b_l, b_a = (_rolling_baseline(submit), _rolling_baseline(lat),
                         _rolling_baseline(aband))
    else:
        b_s, b_l, b_a = submit[0], lat[0], aband[0]
    g_lam = submit / b_s
    g_L = lat / b_l
    g_alpha = (1.0 + aband) / (1.0 + b_a)
    decoupling = np.maximum(1.0, g_lam / g_L)
    quality = 1.0 + np.maximum(g_alpha - 1.0, 0.0)
    sedi = 1.0 / (quality * decoupling)
    depth = np.array([r["depth_proxy"] for r in rows])
    depth_norm = depth / depth[0]
    return sedi, depth_norm, g_lam, g_L, g_alpha


def make_realdata_plot(rows, sedi_fixed, sedi_roll, depth_norm):
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    t = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    # Panel A: stable review depth -> repo operates below saturation
    ax[0].plot(t, depth_norm, "crimson", lw=1.7, marker="o", ms=3,
               label="review-depth proxy (independent)")
    ax[0].axhline(1.0, color="0.6", ls=":", lw=0.9)
    ax[0].set(xlabel="month index", ylabel="normalised review depth",
              ylim=(0, 2.2),
              title="Depth stable (below saturation)")
    ax[0].legend(frameon=False, fontsize=8)
    # Panel B: fixed-baseline false alarms vs rolling-baseline specificity
    ax[1].plot(t, sedi_fixed, color="0.6", lw=1.4, ls="--",
               label="SEDI, fixed baseline")
    ax[1].plot(t, sedi_roll, color="navy", lw=1.8,
               label="SEDI, rolling baseline")
    ax[1].axhline(0.5, color="crimson", ls=":", lw=1.0)
    ax[1].set(xlabel="month index", ylabel="SEDI", ylim=(0, 1.05),
              title="Specificity on a healthy pipeline")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sedi_realdata.png"), dpi=300)
    plt.close(fig)


def _spec(sedi):
    return {"min": float(sedi.min()), "max": float(sedi.max()),
            "mean": float(sedi.mean()), "median": float(np.median(sedi)),
            "frac_below_0.5": float((sedi < 0.5).mean())}


def pgf_coords(rows, sedi_fixed, sedi_roll, depth_norm):
    t = np.arange(len(rows))
    ds = " ".join(f"({i},{d:.3f})" for i, d in zip(t, depth_norm))
    sf = " ".join(f"({i},{s:.3f})" for i, s in zip(t, sedi_fixed))
    sr = " ".join(f"({i},{s:.3f})" for i, s in zip(t, sedi_roll))
    return {"depth_series": ds, "sedi_fixed_series": sf,
            "sedi_rolling_series": sr}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="microsoft/vscode")
    ap.add_argument("--max-prs", type=int, default=2000)
    ap.add_argument("--min-month", type=int, default=15,
                    help="minimum PRs per month for a month to be used")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN env var (public_repo/read scope).")
    owner, name = args.repo.split("/")

    print(f"Fetching up to {args.max_prs} PRs from {args.repo} ...")
    prs = fetch_prs(owner, name, token, args.max_prs)
    print(f"  fetched {len(prs)} PRs")

    rows = monthly_series(prs, min_prs_per_month=args.min_month)
    print(f"  {len(rows)} usable months  ({rows[0]['month']} -> {rows[-1]['month']})")
    sedi_fix, depth_norm, _, _, _ = compute_sedi_real(rows, mode="fixed")
    sedi_roll, _, _, _, _ = compute_sedi_real(rows, mode="rolling")

    depth = np.array([r["depth_proxy"] for r in rows])
    submit = np.array([r["submit"] for r in rows], dtype=float)
    load_depth_r, load_depth_p = stats.pearsonr(submit, depth)
    spec_fix, spec_roll = _spec(sedi_fix), _spec(sedi_roll)

    print(f"\n  Real-data analysis ({args.repo})")
    print(f"    months                 = {len(rows)}")
    print(f"    depth proxy CV         = {depth.std()/depth.mean():.2f}")
    print(f"    load vs depth (Pearson)= {load_depth_r:+.3f} (p={load_depth_p:.2f})")
    print(f"    SEDI fixed:   median={spec_fix['median']:.2f}  "
          f"frac<0.5={spec_fix['frac_below_0.5']:.2f}")
    print(f"    SEDI rolling: median={spec_roll['median']:.2f}  "
          f"frac<0.5={spec_roll['frac_below_0.5']:.2f}")

    results = {
        "repo": args.repo, "n_prs": len(prs), "n_months": len(rows),
        "window": [rows[0]["month"], rows[-1]["month"]],
        "depth_cv": float(depth.std() / depth.mean()),
        "load_vs_depth_pearson_r": float(load_depth_r),
        "load_vs_depth_p": float(load_depth_p),
        "sedi_fixed_specificity": spec_fix,
        "sedi_rolling_specificity": spec_roll,
        "months": [r["month"] for r in rows],
        "pgfplots": pgf_coords(rows, sedi_fix, sedi_roll, depth_norm),
        "series": rows,
    }
    with open(os.path.join(HERE, "realdata_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Wrote realdata_results.json")

    if not args.no_plot:
        make_realdata_plot(rows, sedi_fix, sedi_roll, depth_norm)
        print(f"  Wrote figures/sedi_realdata.png")


if __name__ == "__main__":
    main()
