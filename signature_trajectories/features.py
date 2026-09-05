"""Stage 2/3: paths -> truncated signatures, plus all baseline representations."""
import json, os, math, numpy as np, iisignature, canon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load():
    recs = [json.loads(l) for l in open(os.path.join(OUT, "steps.jsonl"))]
    text = {json.loads(l)["sim_id"]: json.loads(l)["text"]
            for l in open(os.path.join(OUT, "text.jsonl"))}
    return recs, text


def sig_matrix(recs, vocab, R, depth, with_t, cumulative):
    d = len(canon.CHANNELS) - (0 if with_t else 1)
    rows = []
    for r in recs:
        X = canon.path(r, vocab, R, with_t=with_t, cumulative=cumulative)
        if X.shape[0] < 2:                      # signature needs >= 2 points
            rows.append(np.zeros(iisignature.siglength(d, depth)))
        else:
            rows.append(iisignature.sig(X, depth))
    return np.asarray(rows), d


def bag_of_symbols(recs, vocab):
    B = np.zeros((len(recs), len(vocab)))
    for i, r in enumerate(recs):
        for s in r["steps"]:
            B[i, vocab[canon.symbol(s)]] += 1
    return B


def symbol_docs(recs):
    """Symbol sequence rendered as whitespace tokens for the k-gram baseline."""
    return [" ".join(canon.symbol(s).replace(":", "_") for s in r["steps"]) for r in recs]


def build_all(recs, text, vocab, R, seed=0):
    reps = {}
    n_steps = np.array([[len(r["steps"])] for r in recs], dtype=float)
    reps["length"] = np.hstack([n_steps, np.log1p(n_steps)])
    reps["bag_of_symbols"] = bag_of_symbols(recs, vocab)

    docs = symbol_docs(recs)
    tf = TfidfVectorizer(ngram_range=(1, 3), min_df=5, token_pattern=r"\S+",
                         sublinear_tf=True)
    reps["tfidf_kgram"] = tf.fit_transform(docs)

    txt = [text[r["sim_id"]] for r in recs]
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000,
                         sublinear_tf=True, strip_accents="unicode")
    Xt = tv.fit_transform(txt)
    reps["lsa_text"] = TruncatedSVD(384, random_state=seed).fit_transform(Xt)

    dims = {}
    for depth in (2, 3):
        for tag, kw in [("raw", dict(with_t=True, cumulative=False)),
                        ("cum", dict(with_t=True, cumulative=True))]:
            S, d = sig_matrix(recs, vocab, R, depth, **kw)
            reps[f"sig_{tag}_L{depth}"] = S
            dims[f"sig_{tag}_L{depth}"] = (d, depth, S.shape[1])
    # anti-cancellation ablation: identical construction, monotone index removed
    for depth in (2, 3):
        S, d = sig_matrix(recs, vocab, R, depth, with_t=False, cumulative=False)
        reps[f"sig_raw_L{depth}_no_t"] = S
        dims[f"sig_raw_L{depth}_no_t"] = (d, depth, S.shape[1])

    # signature is reparametrisation-invariant, so the raw-path signature cannot
    # see trajectory length at all. Bolt it on explicitly to separate the two.
    reps["sig_raw_L2_plus_len"] = np.hstack([reps["sig_raw_L2"], reps["length"]])
    reps["bag_plus_sig_raw_L2"] = np.hstack([reps["bag_of_symbols"], reps["sig_raw_L2"]])
    return reps, dims
