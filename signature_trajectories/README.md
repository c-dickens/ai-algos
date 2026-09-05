# Path signatures for LLM agent trajectories — validation study

Does the truncated path signature of a canonicalised agent trajectory carry
behavioural signal that simpler order-free representations do not?

Corpus: the tau2-bench published leaderboard simulations
(`sierra-research/tau2-bench`, `data/tau2/results/final`), restricted to the 12
plain `llm_agent` runs — 4 models x 3 domains x 4 trials over a shared task set.

## Pipeline

| stage | file | output |
|---|---|---|
| 1a stream raw results -> compact step records | `extract.py` | `out/steps.jsonl`, `out/text.jsonl` |
| 1b step -> symbol -> integer, hand-designed `R` | `canon.py` | vocabulary + `R` |
| 2/3 paths -> truncated signatures + baselines | `features.py` | in-memory matrices |
| 4 clustering ARI + grouped-CV AUC | `experiment.py` | `out/results.json`, `out/run.log` |
| 4b paired bootstrap on AUC differences | `compare.py` | stdout |
| 4c order-destroying control | `shuffle_control.py` | `out/shuffle_*.json` |
| 5 weight -> channel attribution | `interpret.py` | stdout |

Findings and the verdict are in [`FINDINGS.md`](FINDINGS.md).

## Reproducing

```
git clone --depth 1 https://github.com/sierra-research/tau2-bench /home/user/sierra-research/tau2-bench
python -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn
.venv/bin/pip install --no-build-isolation iisignature   # its setup.py imports numpy
.venv/bin/python extract.py && .venv/bin/python experiment.py
```
