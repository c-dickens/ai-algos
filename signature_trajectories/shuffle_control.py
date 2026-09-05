"""Order-destroying control. Bag-of-symbols is shuffle-invariant by construction;
the signature is not. If the signature's AUC survives shuffling the step order
within each trajectory, whatever it is reading is not order."""
import json, os, sys, numpy as np, canon, features, experiment

HERE = os.path.dirname(os.path.abspath(__file__))


def main(scope="pooled", n_rep=3):
    recs_all, text = features.load()
    vocab, _ = canon.build_vocab(recs_all)
    R = canon.build_R(vocab)
    recs = recs_all if scope == "pooled" else [r for r in recs_all if r["domain"] == scope]
    y = np.array([r["reward"] for r in recs])
    task = np.array([r["task_id"] for r in recs])
    out = {}
    for depth in (2,):
        S, _ = features.sig_matrix(recs, vocab, R, depth, with_t=True, cumulative=False)
        a, lo, hi, _, _ = experiment.grouped_auc(S, y, task)
        out[f"sig_raw_L{depth}"] = (a, lo, hi)
        print(f"{scope} sig_raw_L{depth}  intact  AUC={a:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
        aucs = []
        for rep in range(n_rep):
            rng = np.random.default_rng(100 + rep)
            sh = []
            for r in recs:
                q = dict(r); st = list(r["steps"]); rng.shuffle(st); q["steps"] = st
                sh.append(q)
            Ss, _ = features.sig_matrix(sh, vocab, R, depth, with_t=True, cumulative=False)
            aa, _, _, _, _ = experiment.grouped_auc(Ss, y, task, n_boot=200)
            aucs.append(aa)
            print(f"   shuffled rep {rep}: AUC={aa:.3f}", flush=True)
        out[f"sig_raw_L{depth}_shuffled"] = (float(np.mean(aucs)), float(np.std(aucs)))
        print(f"{scope} sig_raw_L{depth}  shuffled AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}\n", flush=True)
    json.dump(out, open(os.path.join(HERE, "out", f"shuffle_{scope}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
