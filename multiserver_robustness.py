"""
Queueing robustness the review demanded: human review teams are MULTI-SERVER (c>1) and
real queues have ABANDONMENT (cases time out / auto-merge). We show the paper's two
load-bearing facts survive both.

  R7a  Acceleration-factor invariance to server count. The Kingman/Sakasegawa variability
       factor Phi=(Ca^2+Cs^2)/2 multiplies the M/G/c wait identically for any c. A direct
       M/G/c event simulation confirms Wq(log-normal)/Wq(exponential) ~ Phi=1.345 for
       c in {1,2,5}, so the heavy-traffic penalty is NOT an artefact of c=1.
  R7b  Depth-feedback stability is a PER-SERVER-utilisation condition rho^eff=rho/D(rho)<1,
       hence the threshold rho_max solving rho*exp(k(rho-rho_c))=1 is c-invariant. Adding
       reviewers raises absolute capacity c*mu but not the per-server stability margin, so
       oversight saturation is not load-balanced away (cf. Assumption 2).
  R7c  Abandonment (Erlang-A, M/M/c+M). With reneging the queue no longer blows up, but
       oversight degrades through a SECOND channel: the abandoned fraction rises with
       offered load, and abandoned cases are resolved with ZERO review depth. We report
       P(abandon) vs load; saturation reappears as abandonment-driven depth loss.

All seeds fixed. Writes multiserver_results.json + fig_multiserver_robust.png.
"""
import json, math, heapq
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = {}
FIG = "../figures"

def lognormal_sampler(mean, cs2, rng):
    sigma = math.sqrt(math.log(1 + cs2)); mu = math.log(mean) - sigma**2 / 2
    return lambda: rng.lognormal(mu, sigma)

def exp_sampler(mean, rng):
    return lambda: rng.exponential(mean)

def mgc_sim(lam, c, mean_svc, service_sampler, sim_time=400000, warmup=40000, seed=1):
    """M/G/c FCFS. Returns time-average number WAITING (excludes those in service)."""
    rng = np.random.default_rng(seed)
    svc = service_sampler(mean_svc, rng)
    t = 0.0; next_arr = rng.exponential(1 / lam)
    waitq = 0; busy = 0
    dep = []                      # heap of departure times
    area = 0.0; last = 0.0
    while t < sim_time:
        next_dep = dep[0] if dep else math.inf
        if next_arr <= next_dep:
            tn = next_arr; ev = "a"
        else:
            tn = next_dep; ev = "d"
        if tn > warmup:
            area += waitq * (tn - max(last, warmup))
        last = tn; t = tn
        if ev == "a":
            next_arr = t + rng.exponential(1 / lam)
            if busy < c:
                busy += 1; heapq.heappush(dep, t + svc())
            else:
                waitq += 1
        else:
            heapq.heappop(dep)
            if waitq > 0:
                waitq -= 1; heapq.heappush(dep, t + svc())
            else:
                busy -= 1
    return area / (sim_time - warmup)

def R7a_phi_invariance():
    rho = 0.85; cs2 = 1.69; Phi = (1 + cs2) / 2
    res = {}
    for c in [1, 2, 5]:
        mu = 1.0; lam = rho * c * mu; mean = 1 / mu
        wq_ln = mgc_sim(lam, c, mean, lambda m, r: lognormal_sampler(m, cs2, r),
                        seed=10 + c)
        wq_ex = mgc_sim(lam, c, mean, lambda m, r: exp_sampler(m, r), seed=20 + c)
        res[f"c={c}"] = dict(Lq_lognormal=round(float(wq_ln), 3),
                             Lq_exponential=round(float(wq_ex), 3),
                             ratio=round(float(wq_ln / wq_ex), 3))
    OUT["R7a_phi_invariance"] = dict(rho=rho, cs2=cs2, Phi_predicted=round(Phi, 3),
                                     by_c=res,
                                     note="Wq(log-normal)/Wq(exp) ~ Phi for every c: the "
                                          "(Ca^2+Cs^2)/2 variability penalty is server-count "
                                          "invariant, so Observation 1 is not a c=1 artefact.")
    return OUT["R7a_phi_invariance"]

def R7b_stability_invariance():
    k, rho_c = 3.5, 0.60
    f = lambda rho: rho * math.exp(k * max(0.0, rho - rho_c)) - 1.0
    lo, hi = rho_c, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0: hi = mid
        else: lo = mid
    rho_max = (lo + hi) / 2
    OUT["R7b_stability_invariance"] = dict(
        rho_max=round(rho_max, 4),
        note="rho^eff=rho/D(rho) with rho the PER-SERVER utilisation Lambda/(c*mu0); the "
             "stability boundary rho*exp(k(rho-rho_c))=1 contains no c, so rho_max~0.70 "
             "holds for all c. Extra reviewers raise c*mu0 (absolute capacity) but each "
             "server still saturates at the same per-server load, so multi-server staffing "
             "does not by itself escape Assumption 2's sublinear-capacity deficit.")
    return OUT["R7b_stability_invariance"]

def erlang_a_sim(lam, c, mu, theta, sim_time=300000, warmup=30000, seed=1):
    """M/M/c+M (Erlang-A): Poisson arrivals, exp service rate mu/server, exp patience
    rate theta. Returns fraction of arrivals that abandon before service."""
    rng = np.random.default_rng(seed)
    t = 0.0; next_arr = rng.exponential(1 / lam)
    in_service = []                 # heap of service-completion times
    waiting = []                    # heap of (patience-deadline) for queued customers
    arrivals = 0; abandoned = 0
    while t < sim_time:
        nd_s = in_service[0] if in_service else math.inf
        nd_w = waiting[0] if waiting else math.inf
        tn = min(next_arr, nd_s, nd_w)
        counting = tn > warmup
        if tn == next_arr:
            t = tn; next_arr = t + rng.exponential(1 / lam)
            if counting: arrivals += 1
            if len(in_service) < c:
                heapq.heappush(in_service, t + rng.exponential(1 / mu))
            else:
                heapq.heappush(waiting, t + rng.exponential(1 / theta))
        elif tn == nd_s:
            t = tn; heapq.heappop(in_service)
            if waiting:
                heapq.heappop(waiting)        # next waiting enters service
                heapq.heappush(in_service, t + rng.exponential(1 / mu))
        else:
            t = tn; heapq.heappop(waiting)
            if counting: abandoned += 1
    return abandoned / max(1, arrivals)

def R7c_abandonment():
    c = 5; mu = 1.0; theta = 0.5         # patience ~ 2 service times
    loads = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]   # offered load a = lam/(c*mu)
    res = {}
    for a in loads:
        lam = a * c * mu
        pab = erlang_a_sim(lam, c, mu, theta, seed=int(100 * a) + 1)
        res[f"a={a}"] = round(float(pab), 3)
    OUT["R7c_abandonment"] = dict(c=c, patience_rate=theta, abandon_by_offered_load=res,
        note="With reneging the queue is always stable, but the abandoned fraction climbs "
             "with offered load; abandoned cases exit with zero human review. Oversight "
             "saturation thus persists under abandonment as depth loss by attrition rather "
             "than unbounded delay -- the failure mode changes form, not existence.")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    cs = OUT["R7a_phi_invariance"]["by_c"]
    cc = [1, 2, 5]; ratios = [cs[f"c={c}"]["ratio"] for c in cc]
    ax[0].plot(cc, ratios, "o-", color="#0077bb")
    ax[0].axhline(OUT["R7a_phi_invariance"]["Phi_predicted"], color="#cc3311", ls="--",
                  label=f"$\\Phi$={OUT['R7a_phi_invariance']['Phi_predicted']}")
    ax[0].set_xlabel("servers $c$"); ax[0].set_ylabel(r"$L_q$ log-normal / $L_q$ exp")
    ax[0].set_title("R7a: acceleration factor is $c$-invariant"); ax[0].legend(fontsize=8)
    ax[0].set_ylim(1.0, 1.6)
    aa = loads; pab = [res[f"a={a}"] for a in aa]
    ax[1].plot(aa, pab, "s-", color="#ee7733")
    ax[1].axvline(1.0, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("offered load $a=\\lambda/(c\\mu)$"); ax[1].set_ylabel("P(abandon, unreviewed)")
    ax[1].set_title("R7c: abandonment-driven depth loss")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_multiserver_robust.png", dpi=150); plt.close(fig)
    return OUT["R7c_abandonment"]

if __name__ == "__main__":
    print("R7a Phi invariance (M/G/c sim) ...");  R7a_phi_invariance()
    print("R7b stability invariance ...");        R7b_stability_invariance()
    print("R7c abandonment (Erlang-A) ...");      R7c_abandonment()
    with open("multiserver_results.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print(json.dumps(OUT, indent=2))
