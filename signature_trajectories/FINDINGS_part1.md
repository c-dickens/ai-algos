# Path signatures for LLM agent trajectories — a validation study

**Verdict up front.** Cluster structure exists, but the signature does not find
it: on the task-ID partition a plain bag-of-symbols beats every signature
variant by 3–7x adjusted Rand index, and the only partition signatures recover
better than counts is *model ID*, which the ablation in §7 shows is verbosity,
not behaviour. For predicting task success the signature never separates from
the baselines outside bootstrap noise. Numbers in §4–§8.

---

## 0. Corpus

`sierra-research/tau2-bench`, `data/tau2/results/final` — the published
leaderboard simulations. 12 result files kept: 4 models
(`claude-3-7-sonnet-20250219`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`) x 3 domains
(airline, retail, telecom) x 4 trials, **4448 simulations, 205 721 steps**.

This gives exactly the two ground-truth partitions the study needs: every model
attempts the same task set, so *task ID* and *model ID* are crossed rather than
nested, and `reward_info.reward` is programmatic and exactly binary (2528 pass /
1920 fail, 56.8%).

**Excluded, deliberately.** The `telecom-workflow` domain and the
`llm_agent_solo` / `llm_agent_gt` ablation runs are in the same directory. They
change the *scaffold* (no user simulator, oracle policy) rather than the model.
Including them would have doubled the corpus while making "model ID" a
confounded label — a cluster separating `_solo` from `_default` would look like
a model effect and be nothing of the sort.

| domain | tasks | sims | median steps | p95 | pass rate |
|---|---:|---:|---:|---:|---:|
| airline | 50 | 800 | 28 | 55 | 0.539 |
| retail | 114 | 1824 | 30 | 46 | 0.726 |
| telecom | 114 | 1824 | 65 | 117 | 0.542 |

**The primary analysis is within-domain.** Pooling all three would inflate the
task-ID ARI for free, because domains have almost disjoint tool vocabularies and
any representation can separate them perfectly. The pooled row is reported for
completeness, not as the headline.

---

## 1. Canonicalisation

`symbol(step)` drops every ID, timestamp and free-text payload; continuous
quantities survive only as log-scale bucket indices.

```
call:{a|u}:{tool}:a{0-3}      arg-size bucket, edges 8 / 32 / 128 chars
res:{a|u}:{tool}:{ok|err}
say:{agent|user}:l{0-3}       turn-length bucket, edges 64 / 192 / 512 chars
```

The `{a|u}` requestor prefix matters: in telecom the *user simulator* also holds
tools (21 942 of 60 177 calls), and a user-side `reboot_device` is a different
event from an agent-side one.

**Vocabulary — no re-spec needed.**

| scope | V | occurrences | singletons | count<=5 | mass in count<=20 tail | mass in top-50 | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| airline | 50 | 23 145 | 5 | 7 | 0.44% | 100% | 4.42 bits |
| retail | 54 | 56 215 | 1 | 4 | 0.11% | 100% | 4.46 bits |
| telecom | 138 | 126 361 | 12 | 30 | 0.15% | 91.5% | 5.33 bits |
| **pooled** | **214** | 205 721 | 18 | 41 | 0.16% | 85.2% | 5.68 bits |

V is in the low hundreds, not the thousands, and the tail is negligible: the 61
pooled symbols occurring 20 times or fewer account for 0.16% of all occurrences.
The rank-frequency decile histogram is in `out/vocab_report.txt`; the top 20
symbols carry 68% of the mass, and the top 10 are dominated by `say:*` turns and
the read-heavy lookups (`get_details_by_id`, `get_order_details`,
`get_customer_by_phone`).

Design choice worth naming: the arg-size bucket is folded **into the symbol
string** rather than carried alongside it. That is what lets `R` be a genuine
`V x k` lookup with an `arg_size_bucket` column, at the cost of multiplying the
call alphabet by 4. Given V=214, that cost was affordable; had it pushed V into
the thousands the fix would have been to drop the bucket from the symbol and
carry it as a per-step channel instead.

---

## 2. The (n, d) path

`d = 11` channels.

| block | channel | source |
|---|---|---|
| `R` lookup (V x 5) | `is_read`, `is_write`, `is_search`, `is_error`, `arg_size_bucket` | hand-assigned from tool-name semantics |
| contextual (3) | `retry_flag`, `target_novel`, `depth` | computed per step from trace history |
| extra (3) | `t_index`, `cum_err_rate`, `log_out_size` | per the spec |

**`R` cannot be a pure `V x 8` lookup, and this is not a detail.** Four of the
eight requested columns are intrinsic to a symbol and a fifth
(`arg_size_bucket`) is made intrinsic by the choice above. The other three are
not functions of the symbol at all:

- `retry_flag` — has this `(tool, target)` pair already occurred in *this* trace?
- `target_novel` — is this argument-value set new to this trace?
- `depth` — how many tool steps deep is the agent since it last spoke, `log1p`
  scaled? (Read as autonomy: how long a chain it runs without checking in. The
  alternative reading, position within a parallel call batch, is a scaffold
  artefact rather than a behaviour, so it was not used.)

All three are history-dependent, so they are computed per step. `R` is honestly
a `V x 5` table and the symbol block is 5 lookup + 3 contextual columns. Note
that `retry_flag` and `target_novel` are computed from the *un*-stripped
argument values — IDs are exactly what identifies a target, so ID-stripping
applies to the symbol alphabet, not to the novelty bookkeeping.

`is_search` is hand-separated from `is_read` as *lookup-by-attribute*
(`find_user_id_by_name_zip`, `get_customer_by_phone`, `search_direct_flight`)
versus reading an entity you already have an ID for. That distinction is the
whole point of having an interpretable table rather than a random Gaussian one.

**Channel scaling.** Level-k signature terms scale like (channel range)^k, so a
channel with a range of 9 swamps one with a range of 1 by 9^k at level k.
`log_out_size` is therefore divided by 10 and `depth` normalised, putting every
channel in roughly [0, 1.5].

**Raw versus cumulative lift — the choice that mattered most.** `E = R[s]` can
be fed to the signature either as the path points directly, or accumulated. Both
are reported throughout, because they are different objects:

- `sig_raw` — path points are `R[s_i]`. This is where the anti-cancellation
  concern actually bites, since the path wanders and can retrace.
- `sig_cum` — path points are `cumsum(R[s_i])`. Every symbol-block column is
  non-negative, so the path is monotone in each coordinate and *cannot* be
  tree-like. Its level-1 terms are exactly the channel totals — i.e. level 1 of
  the cumulative signature **is** a projection of bag-of-symbols, which makes
  the "does order beat counts" question directly readable off the level split.

---

## 3. Signatures — dimension checks

`iisignature`, depths 2 and 3.

| d | L | (d^(L+1)-1)/(d-1) | `iisignature.siglength` |
|---:|---:|---:|---:|
| 11 | 2 | 133 | 132 |
| 11 | 3 | 1464 | 1463 |
| 10 (no `t`) | 2 | 111 | 110 |
| 10 (no `t`) | 3 | 1111 | 1110 |

The formula holds exactly; `iisignature` omits the level-0 term, which is
identically 1, hence the constant off-by-one. Output dimension is identical
across trajectory lengths — verified on paths of n=6 and n=296, both giving 1463
coordinates at L=3.

**A property that turns out to drive the whole result.** The signature is
invariant to reparametrisation: resampling a path to 69 points along the same
piecewise-linear image as its 35-point original changes the depth-3 signature by
5e-16. Since `t_index` runs 0 -> 1 regardless of n, **the raw-path signature
cannot see trajectory length at all**. That matters here because length is the
single most predictive scalar in this corpus — point-biserial correlation with
reward of -0.34 pooled, -0.37 on telecom (failing traces average 57.9 steps
against 37.4 for passing ones). The representation is structurally blind to the
strongest available signal. `sig_raw_L2_plus_len` and the cumulative variant
(whose level-1 terms are counts, and therefore do see length) are included to
separate "the signature adds nothing" from "the signature cannot see length".
