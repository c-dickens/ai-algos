"""Stage 4/5: clustering ARI against the two ground-truth partitions, and
grouped-CV logistic regression for task success, for every representation."""
import json, os, sys, time, numpy as np, scipy.sparse as sp
import canon, features
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 0
CS = [0.01, 0.1, 1.0, 10.0]
HERE = os.path.dirname(os.path.abspath(__file__))


def scale(X, fit_idx=None):
    """z-score (max-abs for sparse) fit on `fit_idx` only. Signature coordinates
    at higher levels span ~30 orders of magnitude and a few are numerically dead;
    columns with negligible variance are left alone rather than amplified."""
    idx = slice(None) if fit_idx is None else fit_idx
    if sp.issparse(X):
        return MaxAbsScaler().fit(X[idx]).transform(X)
    s = StandardScaler().fit(X[idx])
    s.scale_ = np.where(s.var_ < 1e-12 * max(s.var_.max(), 1e-300), 1.0, s.scale_)
    return np.nan_to_num(s.transform(X), nan=0.0, posinf=0.0, neginf=0.0)


def rowspace(Z, tr):
    """L2-penalised logistic regression depends on X only through the row space
    of the training block, and the L2 penalty is rotation invariant. So when
    p > n_train we can project onto that row space with no change to the fitted
    model or its test-set predictions -- just far cheaper. Verified identical to
    4 d.p. against the full-dimensional fit."""
    if sp.issparse(Z) or Z.shape[1] <= len(tr):
        return Z
    _, sv, Vt = np.linalg.svd(Z[tr], full_matrices=False)
    r = int((sv > 1e-9 * sv[0]).sum())
    return Z @ Vt[:r].T


def cluster_scores(X, labels_task, labels_model, labels_trial, n_pca=50):
    Xs = scale(X)
    if sp.issparse(Xs):
        Xs = Xs.toarray()
    out = {}
    for tag, Z in [("raw", Xs),
                   ("pca50", PCA(min(n_pca, *Xs.shape), random_state=SEED).fit_transform(Xs))]:
        for name, y in [("task", labels_task), ("model", labels_model), ("trial", labels_trial)]:
            k = len(set(y))
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
            out[f"ari_{name}_{tag}"] = adjusted_rand_score(y, km.labels_)
    return out


def grouped_auc(X, y, groups, n_boot=2000):
    """Out-of-fold AUC with nested C selection; CI by task-level cluster
    bootstrap of the out-of-fold predictions."""
    oof = np.zeros(len(y))
    outer = GroupKFold(n_splits=5)
    picked = []
    for tr, te in outer.split(X, y, groups):
        Xs = rowspace(scale(X, tr), tr)
        best, best_auc = CS[0], -1
        inner = GroupKFold(n_splits=3)
        for C in CS:
            preds = np.zeros(len(tr))
            for itr, ite in inner.split(Xs[tr], y[tr], groups[tr]):
                m = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr][itr], y[tr][itr])
                preds[ite] = m.predict_proba(Xs[tr][ite])[:, 1]
            a = roc_auc_score(y[tr], preds)
            if a > best_auc:
                best_auc, best = a, C
        picked.append(best)
        m = LogisticRegression(C=best, max_iter=2000).fit(Xs[tr], y[tr])
        oof[te] = m.predict_proba(Xs[te])[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    boots = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[ii])) < 2:
            continue
        boots.append(roc_auc_score(y[ii], oof[ii]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi, picked, oof


def main(scopes):
    recs_all, text = features.load()
    vocab, counts = canon.build_vocab(recs_all)
    R = canon.build_R(vocab)
    results = {}
    for scope in scopes:
        recs = recs_all if scope == "pooled" else [r for r in recs_all if r["domain"] == scope]
        y = np.array([r["reward"] for r in recs])
        task = np.array([r["task_id"] for r in recs])
        model = np.array([r["model"] for r in recs])
        trial = np.array([str(r["trial"]) for r in recs])
        t0 = time.time()
        reps, dims = features.build_all(recs, text, vocab, R, seed=SEED)
        print(f"[{scope}] n={len(recs)} tasks={len(set(task))} pass={y.mean():.3f} "
              f"feats built in {time.time()-t0:.0f}s", flush=True)
        res = {"n": len(recs), "n_tasks": int(len(set(task))), "pass_rate": float(y.mean()),
               "dims": {k: list(v) for k, v in dims.items()}, "reps": {}}
        for name, X in reps.items():
            t1 = time.time()
            row = {"p": int(X.shape[1])}
            row.update(cluster_scores(X, task, model, trial))
            auc, lo, hi, picked, oof = grouped_auc(X, y, task)
            row.update(auc=auc, auc_lo=lo, auc_hi=hi, C=picked)
            res["reps"][name] = row
            np.save(os.path.join(HERE, "out", f"oof_{scope}_{name}.npy"), oof)
            print(f"   {name:24s} p={X.shape[1]:6d}  ARI task={row['ari_task_pca50']:.3f} "
                  f"model={row['ari_model_pca50']:.3f}  AUC={auc:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"({time.time()-t1:.0f}s)", flush=True)
        results[scope] = res
        json.dump(results, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:] or ["airline", "retail", "telecom", "pooled"])
