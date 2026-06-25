import os as _os; _os.makedirs("figures", exist_ok=True)
"""
Multi-year monthly SEDI time series from the GitHub SEARCH API (date-bucketed), to hunt
for a REAL saturation episode in a high-growth review pipeline. One search call per month
gives total_count (submissions) plus a <=100-PR sample with latency, abandonment, and a
depth proxy (comments per merged PR). Throttled to 10/min (in-process sleep) and cached per
month so re-runs cost nothing. INTEGRITY: depth here is a measured proxy, not human coding;
episodes are reported as detected, not asserted.
"""
import os, sys, json, time, calendar, urllib.request, urllib.error
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CACHE = "/tmp/gh_search"; os.makedirs(CACHE, exist_ok=True)
FIG = "figures"; TOKEN = os.environ.get("GH_TOKEN", "")

def search_month(repo, year, month):
    last = calendar.monthrange(year, month)[1]
    q = f"repo:{repo}+type:pr+created:{year}-{month:02d}-01..{year}-{month:02d}-{last:02d}"
    url = f"https://api.github.com/search/issues?q={q}&per_page=100"
    key = os.path.join(CACHE, f"{repo.replace('/','_')}_{year}-{month:02d}.json")
    if os.path.exists(key):
        return json.load(open(key))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "sedi"})
    if TOKEN: req.add_header("Authorization", f"Bearer {TOKEN}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
            json.dump(d, open(key, "w")); time.sleep(6.5); return d
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print(f"  rate-limited, waiting 30s ({year}-{month:02d})"); time.sleep(30); continue
            print(f"  HTTP {e.code} {year}-{month:02d}"); return None
    return None

def month_metrics(d):
    if not d or "items" not in d: return None
    items = d["items"]; merged_lat = []; comments = []; merged = 0; closed_unmerged = 0
    for it in items:
        pr = it.get("pull_request") or {}; ma = pr.get("merged_at")
        c = datetime.fromisoformat(it["created_at"].replace("Z", "+00:00"))
        if ma:
            merged += 1
            md = datetime.fromisoformat(ma.replace("Z", "+00:00"))
            merged_lat.append((md - c).total_seconds() / 3600.0)
            comments.append(it.get("comments", 0))
        elif it.get("state") == "closed":
            closed_unmerged += 1
    if merged < 3: return None
    return dict(submissions=d.get("total_count", len(items)),
                median_hours=float(np.median(merged_lat)),
                abandon=closed_unmerged / max(1, closed_unmerged + merged),
                depth_proxy=float(np.mean(comments)))

def gen_months(y0, m0, y1, m1):
    out = []; y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m)); m += 1
        if m > 12: m = 1; y += 1
    return out

def build(repo, y0, m0, y1, m1):
    months = gen_months(y0, m0, y1, m1); rows = []
    for (y, m) in months:
        d = search_month(repo, y, m); mm = month_metrics(d)
        if mm: mm["month"] = f"{y}-{m:02d}"; rows.append(mm)
    return rows

def sedi_variants(rows):
    gl = np.array([r["submissions"] for r in rows], float)
    gL = np.array([r["median_hours"] for r in rows], float)
    ga = np.array([r["abandon"] for r in rows], float) + 1e-6
    def roll(x, w=6):
        o = np.ones(len(x))
        for i in range(len(x)):
            lo = max(0, i-w); b = slice(lo, i) if i > lo else slice(0, 1)
            o[i] = x[i]/np.mean(x[b])
        return o
    glr, gLr, gar = roll(gl), roll(gL), roll(ga)
    glf, gLf, gaf = gl/gl[0], gL/gL[0], ga/ga[0]
    base_ga = np.mean(ga[:min(6, len(ga))])
    core = lambda dec, q: 1.0/(np.maximum(1.0, dec)*(1.0+np.maximum(0.0, q-1.0)))
    return dict(fixed=core(glf/gLf, gaf), rolling=core(glr/gLr, gar),
                hybrid=core(glr/gLr, ga/base_ga), submissions=gl)

def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "huggingface/transformers"
    y0, m0 = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2020-09").split("-"))
    y1, m1 = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2023-12").split("-"))
    rows = build(repo, y0, m0, y1, m1)
    print(f"{repo}: {len(rows)} usable months "
          f"({rows[0]['month'] if rows else '-'}..{rows[-1]['month'] if rows else '-'})")
    if len(rows) < 8: print("insufficient"); return
    sv = sedi_variants(rows); depth = np.array([r["depth_proxy"] for r in rows])
    months = [r["month"] for r in rows]
    r_hyb = stats.pearsonr(sv["hybrid"], depth)[0]
    # episode: hybrid SEDI below threshold AND depth proxy below its median
    thr = 0.5; dip = (sv["hybrid"] < thr) & (depth < np.median(depth))
    res = dict(repo=repo, window=f"{months[0]}..{months[-1]}", n_months=len(rows),
               submissions_growth=round(float(sv["submissions"][-1]/sv["submissions"][0]), 2),
               depth_cv=round(float(np.std(depth)/np.mean(depth)), 3),
               pearson_hybrid_vs_depth=round(float(r_hyb), 3),
               hybrid_alarm_months=[months[i] for i in np.where(dip)[0]],
               min_hybrid=round(float(np.min(sv["hybrid"])), 3),
               min_hybrid_month=months[int(np.argmin(sv["hybrid"]))])
    json.dump(dict(result=res, months=months,
                   submissions=sv["submissions"].tolist(),
                   median_hours=[r["median_hours"] for r in rows],
                   abandon=[r["abandon"] for r in rows],
                   depth_proxy=depth.tolist(),
                   sedi_hybrid=sv["hybrid"].tolist(), sedi_fixed=sv["fixed"].tolist()),
              open(f"timeseries_{repo.replace('/','_')}.json", "w"), indent=2)
    # figure
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(10, 4)); ax2 = ax.twinx()
    ax.bar(x, sv["submissions"]/sv["submissions"].max(), color="#dddddd", label="submissions (norm)")
    ax.plot(x, depth/np.max(depth), color="#cc3311", label="depth proxy (norm)")
    ax.plot(x, sv["hybrid"], color="#0077bb", lw=2, label="SEDI hybrid")
    ax.plot(x, sv["fixed"], color="#999999", ls="--", label="SEDI fixed")
    ax.axhline(thr, color="k", ls=":", lw=0.8)
    ax2.plot(x, [r["median_hours"] for r in rows], color="#009988", alpha=0.5, label="median hrs-to-merge")
    ax.set_xticks(x[::3]); ax.set_xticklabels(months[::3], rotation=45, fontsize=7)
    ax.set_ylim(0, 1.3); ax.set_ylabel("normalised / SEDI"); ax2.set_ylabel("median hours")
    ax.set_title(f"{repo}: real monthly SEDI ({res['window']})")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_timeseries.png", dpi=300); plt.close(fig)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
