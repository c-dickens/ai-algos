"""Stage 5: decode the largest logistic-regression weights back to channels.

iisignature layout for depth L over d channels (level-0 omitted):
  [0, d)            level 1, coordinate i          -> total increment of channel i
  [d, d + d^2)      level 2, row-major (i, j)      -> ordered integral, i before j
  ...
Level-2 antisymmetric part (S^ij - S^ji)/2 is the signed Levy area of channels
(i, j): positive means i tends to move before j.
"""
import json, os, sys, numpy as np, canon, features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
CH = canon.CHANNELS


def decode(idx, d):
    if idx < d:
        return 1, (idx,)
    idx -= d
    if idx < d * d:
        return 2, (idx // d, idx % d)
    idx -= d * d
    return 3, (idx // (d * d), (idx // d) % d, idx % d)


def name(t):
    return " -> ".join(CH[i] for i in t)


def main(scope="pooled", rep="sig_raw_L2", top=20):
    top = int(top)
    recs_all, text = features.load()
    vocab, _ = canon.build_vocab(recs_all)
    R = canon.build_R(vocab)
    recs = recs_all if scope == "pooled" else [r for r in recs_all if r["domain"] == scope]
    y = np.array([r["reward"] for r in recs])
    res = json.load(open(os.path.join(HERE, "out", "results.json")))
    C = float(np.median(res[scope]["reps"][rep]["C"]))
    d = len(CH)
    S, _ = features.sig_matrix(recs, vocab, R, int(rep[-1]), with_t=True,
                               cumulative=rep.startswith("sig_cum"))
    Z = np.nan_to_num(StandardScaler().fit_transform(S))
    m = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    w = m.coef_[0]
    order = np.argsort(-np.abs(w))
    print(f"=== {scope} / {rep}  (C={C}, standardised coefficients) ===")
    print(f"{'rank':>4} {'coef':>8} {'lvl':>3}  channels")
    for r, i in enumerate(order[:top]):
        lvl, t = decode(int(i), d)
        print(f"{r+1:4d} {w[i]:+8.3f} {lvl:3d}  {name(t)}")
    lvl_mass = {}
    for i, wi in enumerate(w):
        lvl, _ = decode(i, d)
        lvl_mass[lvl] = lvl_mass.get(lvl, 0) + abs(wi)
    tot = sum(lvl_mass.values())
    print("\n|weight| mass by signature level:",
          {k: f"{v/tot:.3f}" for k, v in sorted(lvl_mass.items())})
    # channel-level attribution: how much |weight| touches each channel
    ch_mass = np.zeros(d)
    for i, wi in enumerate(w):
        _, t = decode(i, d)
        for c in t:
            ch_mass[c] += abs(wi)
    print("\n|weight| mass by channel:")
    for c in np.argsort(-ch_mass):
        print(f"   {CH[c]:16s} {ch_mass[c]/ch_mass.sum():.3f}")
    # Levy areas: antisymmetric level-2 part, most discriminative ordered pairs
    print("\ntop signed-area (Levy) pairs by |w_ij - w_ji|:")
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            a, b = d + i * d + j, d + j * d + i
            pairs.append((abs(w[a] - w[b]), w[a] - w[b], i, j))
    for mag, val, i, j in sorted(pairs, reverse=True)[:10]:
        print(f"   {val:+7.3f}  ({CH[i]}, {CH[j]})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
