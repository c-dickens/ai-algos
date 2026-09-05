"""Is the model-ID cluster signal behavioural, or just verbosity?

The symbol alphabet buckets natural-language turn length (say:agent:l0..l3).
Response length is about the most model-characteristic thing there is and has
nothing to do with tool-use behaviour. Collapse those buckets and re-measure.
"""
import json, os, sys, numpy as np, canon, features, experiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = ["sig_raw_L2", "sig_cum_L2", "bag_of_symbols", "tfidf_kgram"]


def main(scope="pooled"):
    recs_all, text = features.load()
    recs = recs_all if scope == "pooled" else [r for r in recs_all if r["domain"] == scope]
    task = np.array([r["task_id"] for r in recs])
    model = np.array([r["model"] for r in recs])
    trial = np.array([str(r["trial"]) for r in recs])
    y = np.array([r["reward"] for r in recs])
    out = {}
    orig = canon.symbol
    for tag, fn in [("with_verbosity", orig),
                    ("no_verbosity",
                     lambda s: orig(s).rsplit(":", 1)[0] if s["kind"] == "say" else orig(s))]:
        canon.symbol = fn
        vocab, _ = canon.build_vocab(recs_all)
        R = canon.build_R(vocab)
        reps = {
            "bag_of_symbols": features.bag_of_symbols(recs, vocab),
            "tfidf_kgram": TfidfVectorizer(ngram_range=(1, 3), min_df=5,
                                           token_pattern=r"\S+", sublinear_tf=True
                                           ).fit_transform(features.symbol_docs(recs)),
            "sig_raw_L2": features.sig_matrix(recs, vocab, R, 2, with_t=True,
                                              cumulative=False)[0],
            "sig_cum_L2": features.sig_matrix(recs, vocab, R, 2, with_t=True,
                                              cumulative=True)[0],
        }
        print(f"--- {scope} / {tag} (V={len(vocab)}) ---", flush=True)
        for name in REPS:
            X = reps[name]
            sc = experiment.cluster_scores(X, task, model, trial)
            auc, lo, hi, _, _ = experiment.grouped_auc(X, y, task, n_boot=500)
            out[f"{tag}|{name}"] = dict(sc, auc=auc, auc_lo=lo, auc_hi=hi)
            print(f"   {name:18s} ARI model={sc['ari_model_pca50']:.3f} "
                  f"task={sc['ari_task_pca50']:.3f}  AUC={auc:.3f}", flush=True)
    canon.symbol = orig
    json.dump(out, open(os.path.join(HERE, "out", f"ablate_verbosity_{scope}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
