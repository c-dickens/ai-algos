"""Render out/results.json as markdown tables."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["length", "bag_of_symbols", "tfidf_kgram", "lsa_text",
         "sig_raw_L2", "sig_raw_L3", "sig_cum_L2", "sig_cum_L3",
         "sig_raw_L2_no_t", "sig_raw_L3_no_t",
         "sig_raw_L2_plus_len", "bag_plus_sig_raw_L2"]
LABEL = {"length": "trajectory length", "bag_of_symbols": "bag-of-symbols",
         "tfidf_kgram": "TF-IDF symbol 1-3grams", "lsa_text": "LSA text (embedding stand-in)",
         "sig_raw_L2": "signature raw L=2", "sig_raw_L3": "signature raw L=3",
         "sig_cum_L2": "signature cumulative L=2", "sig_cum_L3": "signature cumulative L=3",
         "sig_raw_L2_no_t": "signature raw L=2, no t", "sig_raw_L3_no_t": "signature raw L=3, no t",
         "sig_raw_L2_plus_len": "signature raw L=2 + length",
         "bag_plus_sig_raw_L2": "bag-of-symbols + signature raw L=2"}


def main(path=None):
    res = json.load(open(path or os.path.join(HERE, "out", "results.json")))
    for scope, r in res.items():
        print(f"\n### {scope}  (n={r['n']}, {r['n_tasks']} tasks, pass rate {r['pass_rate']:.3f})\n")
        print("| representation | p | ARI task | ARI model | ARI trial | AUC success | 95% CI |")
        print("|---|---:|---:|---:|---:|---:|---|")
        for k in ORDER:
            if k not in r["reps"]:
                continue
            v = r["reps"][k]
            print(f"| {LABEL[k]} | {v['p']} | {v['ari_task_pca50']:.3f} | "
                  f"{v['ari_model_pca50']:.3f} | {v['ari_trial_pca50']:.3f} | "
                  f"**{v['auc']:.3f}** | [{v['auc_lo']:.3f}, {v['auc_hi']:.3f}] |")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
