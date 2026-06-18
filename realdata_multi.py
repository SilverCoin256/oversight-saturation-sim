import os as _os; _os.makedirs("figures", exist_ok=True)
"""
Real multi-repository SEDI study (Plan C 'second dataset' + the observational part of
Plan D). Pulls REAL pull-request telemetry from the public GitHub API, computes the three
boundary observables AND an independent depth proxy (review-thread comments per merged PR),
then runs fixed / rolling / hybrid-baseline SEDI and tests for genuine saturation episodes.

INTEGRITY: the depth signal here is a measured PROXY (comments per merged PR), not human-
coded ground-truth review depth. Episodes are reported as detected, not asserted. Pages are
cached to disk so re-runs cost no API calls. Unauthenticated API = 60 req/hr; set GH_TOKEN
to raise the limit.
"""
import os, json, time, math, urllib.request, urllib.error
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CACHE = "/tmp/gh_cache"; os.makedirs(CACHE, exist_ok=True)
FIG = "figures"
TOKEN = os.environ.get("GH_TOKEN", "")

def api_get(url):
    cache_key = os.path.join(CACHE, url.split("github.com/")[-1].replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_") + ".json")
    if os.path.exists(cache_key):
        return json.load(open(cache_key))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "sedi-study"})
    if TOKEN: req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        json.dump(data, open(cache_key, "w"))
        return data
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {url[:70]}"); return None

def pull_repo(repo, pages=15):
    items = []
    for p in range(1, pages + 1):
        url = (f"https://api.github.com/repos/{repo}/issues"
               f"?state=all&per_page=100&page={p}&sort=created&direction=desc")
        d = api_get(url)
        if not d: break
        prs = [x for x in d if "pull_request" in x]
        items.extend(prs)
        if len(d) < 100: break
    return items

def monthly_series(items):
    by_month = defaultdict(lambda: dict(n=0, lat=[], merged=0, closed=0, comments=[]))
    for it in items:
        try:
            c = datetime.fromisoformat(it["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        key = f"{c.year}-{c.month:02d}"
        m = by_month[key]; m["n"] += 1
        pr = it.get("pull_request") or {}
        merged_at = pr.get("merged_at")
        if merged_at:
            m["merged"] += 1
            md = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            m["lat"].append((md - c).total_seconds() / 3600.0)   # hours to merge
            m["comments"].append(it.get("comments", 0))
        elif it.get("state") == "closed":
            m["closed"] += 1
    months = sorted(by_month)
    out = dict(month=[], g_lambda=[], median_hours=[], abandon=[], depth_proxy=[])
    for k in months:
        m = by_month[k]
        if m["merged"] < 3:   # need a few merges for stable stats
            continue
        out["month"].append(k); out["g_lambda"].append(m["n"])
        out["median_hours"].append(float(np.median(m["lat"])))
        out["abandon"].append(m["closed"] / max(1, m["closed"] + m["merged"]))
        out["depth_proxy"].append(float(np.mean(m["comments"])))
    return out

def baselines(x, window=6):
    x = np.asarray(x, float); roll = np.ones(len(x)); fixed = x / x[0]
    for i in range(len(x)):
        lo = max(0, i - window); b = slice(lo, i) if i > lo else slice(0, 1)
        roll[i] = x[i] / np.mean(x[b])
    return fixed, roll

def sedi_variants(s):
    gl = np.asarray(s["g_lambda"], float); gL = np.asarray(s["median_hours"], float)
    ga = np.asarray(s["abandon"], float) + 1e-6
    glf, glr = baselines(gl); gLf, gLr = baselines(gL); gaf, gar = baselines(ga)
    base_ga = np.mean(ga[:min(6, len(ga))])
    def core(decoup, qual): return 1.0 / (np.maximum(1.0, decoup) * (1.0 + np.maximum(0.0, qual - 1.0)))
    fixed  = core(glf / gLf, gaf)
    rolling= core(glr / gLr, gar)
    hybrid = core(glr / gLr, ga / base_ga)     # rolling decoupling, ABSOLUTE disputes
    return dict(fixed=fixed, rolling=rolling, hybrid=hybrid)

REPOS = ["scikit-learn/scikit-learn", "pandas-dev/pandas", "huggingface/transformers"]

def main():
    rl = api_get("https://api.github.com/rate_limit")
    if rl: print("rate remaining:", rl["resources"]["core"]["remaining"])
    results = {}
    fig, axes = plt.subplots(1, len(REPOS), figsize=(4.3*len(REPOS), 3.6))
    if len(REPOS) == 1: axes = [axes]
    for ax, repo in zip(axes, REPOS):
        print("pulling", repo, "...")
        items = pull_repo(repo, pages=15)
        s = monthly_series(items)
        if len(s["month"]) < 6:
            print("  insufficient months:", len(s["month"])); continue
        sv = sedi_variants(s)
        depth = np.asarray(s["depth_proxy"], float)
        r_fix = stats.pearsonr(sv["fixed"], depth)[0]
        r_hyb = stats.pearsonr(sv["hybrid"], depth)[0]
        thr = 0.5
        fa_fixed = float(np.mean(sv["fixed"] < thr))
        fa_hyb = float(np.mean(sv["hybrid"] < thr))
        # episode flag: a month where hybrid SEDI dips AND depth proxy is below its own median
        dip = (sv["hybrid"] < thr) & (depth < np.median(depth))
        results[repo] = dict(
            window=f'{s["month"][0]}..{s["month"][-1]}', n_months=len(s["month"]),
            depth_cv=round(float(np.std(depth)/np.mean(depth)), 3),
            pearson_hybrid_vs_depth=round(float(r_hyb), 3),
            pearson_fixed_vs_depth=round(float(r_fix), 3),
            fixed_alarm_frac=round(fa_fixed, 3), hybrid_alarm_frac=round(fa_hyb, 3),
            episode_months=[s["month"][i] for i in np.where(dip)[0]])
        x = np.arange(len(s["month"]))
        ax.plot(x, depth/np.max(depth), color="#cc3311", label="depth proxy (norm)")
        ax.plot(x, sv["fixed"], color="#999999", ls="--", label="SEDI fixed")
        ax.plot(x, sv["hybrid"], color="#0077bb", label="SEDI hybrid")
        ax.axhline(thr, color="k", ls=":", lw=0.7)
        ax.set_title(repo.split("/")[-1], fontsize=9); ax.set_ylim(0, 1.25)
        ax.set_xlabel(f'{s["month"][0]} .. {s["month"][-1]}', fontsize=7)
        if ax is axes[0]: ax.legend(fontsize=7)
    fig.suptitle("Real multi-repo SEDI (depth proxy = comments per merged PR)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_multirepo.png", dpi=150); plt.close(fig)
    json.dump(results, open("realdata_multi_results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
