"""Paired task-level cluster bootstrap on the AUC *difference* between two
representations, reusing the saved out-of-fold predictions. Comparing two
marginal CIs is not a test; this is."""
import json, os, sys, numpy as np, features
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))


def load_scope(scope):
    recs_all, _ = features.load()
    recs = recs_all if scope == "pooled" else [r for r in recs_all if r["domain"] == scope]
    y = np.array([r["reward"] for r in recs])
    g = np.array([r["task_id"] for r in recs])
    return y, g


def paired(scope, a, b, n_boot=4000, seed=0):
    y, g = load_scope(scope)
    pa = np.load(os.path.join(HERE, "out", f"oof_{scope}_{a}.npy"))
    pb = np.load(os.path.join(HERE, "out", f"oof_{scope}_{b}.npy"))
    d0 = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    ds = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[u] for u in gs])
        if len(set(y[ii])) < 2:
            continue
        ds.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, lo, hi, max(p, 1.0 / len(ds))


if __name__ == "__main__":
    res = json.load(open(os.path.join(HERE, "out", "results.json")))
    PAIRS = [("sig_raw_L2", "bag_of_symbols"), ("sig_cum_L2", "bag_of_symbols"),
             ("sig_raw_L3", "bag_of_symbols"), ("sig_cum_L3", "bag_of_symbols"),
             ("sig_raw_L2", "length"), ("sig_cum_L2", "length"),
             ("bag_of_symbols", "length"), ("tfidf_kgram", "bag_of_symbols"),
             ("sig_raw_L2_plus_len", "length"),
             ("bag_plus_sig_raw_L2", "bag_of_symbols"),
             ("sig_raw_L2", "sig_raw_L2_no_t"), ("sig_raw_L3", "sig_raw_L3_no_t"),
             ("lsa_text", "bag_of_symbols")]
    for scope in sys.argv[1:] or list(res):
        print(f"\n=== {scope} (n={res[scope]['n']}, {res[scope]['n_tasks']} tasks) ===")
        print(f"{'A':24s} {'vs B':24s} {'dAUC':>7} {'95% CI':>18} {'p':>7}")
        for a, b in PAIRS:
            if a not in res[scope]["reps"] or b not in res[scope]["reps"]:
                continue
            d, lo, hi, p = paired(scope, a, b)
            star = " *" if lo > 0 or hi < 0 else ""
            print(f"{a:24s} {b:24s} {d:+7.3f}  [{lo:+.3f},{hi:+.3f}] {p:7.4f}{star}")
