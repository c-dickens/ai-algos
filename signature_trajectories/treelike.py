"""How much does the anti-cancellation guard actually matter here?

A path that is tree-like (retraces itself exactly) has trivial signature. The
monotone index t is meant to prevent that. Measure directly how close the
un-guarded paths come to degeneracy, and how much of each trace is an exact
retrace of an earlier segment.
"""
import json, os, numpy as np, iisignature, canon

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "out", "steps.jsonl"))]
    vocab, _ = canon.build_vocab(recs)
    R = canon.build_R(vocab)
    n_t, n_no = [], []
    exact_back, tot_seg, zero_inc = 0, 0, 0
    for r in recs:
        Xt = canon.path(r, vocab, R, with_t=True)
        Xn = canon.path(r, vocab, R, with_t=False)
        if len(Xt) < 2:
            continue
        n_t.append(np.linalg.norm(iisignature.sig(Xt, 2)))
        n_no.append(np.linalg.norm(iisignature.sig(Xn, 2)))
        D = np.diff(Xn, axis=0)
        tot_seg += len(D)
        zero_inc += int((np.abs(D).max(axis=1) < 1e-12).sum())
        exact_back += int((np.abs(D[:-1] + D[1:]).max(axis=1) < 1e-12).sum())
    n_t, n_no = np.array(n_t), np.array(n_no)
    print(f"trajectories: {len(n_t)}   un-guarded path segments: {tot_seg}")
    print(f"  zero-increment segments (consecutive identical embedding): "
          f"{zero_inc} ({zero_inc/tot_seg:.3%})")
    print(f"  immediate exact backtracks (d_i = -d_i+1, the tree-like motif): "
          f"{exact_back} ({exact_back/tot_seg:.3%})")
    print(f"\n||sig L=2|| with t   : med {np.median(n_t):.3f}  p1 {np.percentile(n_t,1):.4f}  min {n_t.min():.4f}")
    print(f"||sig L=2|| without t: med {np.median(n_no):.3f}  p1 {np.percentile(n_no,1):.4f}  min {n_no.min():.4f}")
    for thr in (1e-6, 1e-3, 1e-2):
        print(f"  trajectories with ||sig|| < {thr:g} : with t {int((n_t<thr).sum())}, without t {int((n_no<thr).sum())}")
    print(f"\ncorrelation of the two norms: {np.corrcoef(n_t, n_no)[0,1]:.4f}")


if __name__ == "__main__":
    main()
