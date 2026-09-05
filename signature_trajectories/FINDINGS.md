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

---

## 4. Results

Identical pipeline for every representation. Clustering: z-score (max-abs for
sparse) -> PCA to 50 components -> KMeans with `k` set to the *true* number of
classes, which is generous to the method. ARI is reported against three
partitions; `trial` is a negative control that should be 0. Prediction: L2
logistic regression, `GroupKFold(5)` on **task ID** so no task appears in both
train and test, inner `GroupKFold(3)` for `C` in {0.01, 0.1, 1, 10}, AUC on
pooled out-of-fold predictions, 95% CI from a **task-level cluster bootstrap**
(4 trials x 4 models per task are not independent observations).


### airline  (n=800, 50 tasks, pass rate 0.539)

| representation | p | ARI task | ARI model | ARI trial | AUC success | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| trajectory length | 2 | 0.043 | 0.031 | -0.002 | **0.636** | [0.551, 0.717] |
| bag-of-symbols | 214 | 0.257 | 0.021 | -0.002 | **0.607** | [0.504, 0.708] |
| TF-IDF symbol 1-3grams | 865 | 0.200 | 0.265 | -0.003 | **0.672** | [0.568, 0.770] |
| LSA text (embedding stand-in) | 384 | 0.089 | 0.020 | -0.002 | **0.560** | [0.499, 0.621] |
| signature raw L=2 | 132 | 0.079 | 0.240 | -0.001 | **0.622** | [0.525, 0.721] |
| signature raw L=3 | 1463 | 0.068 | 0.201 | -0.001 | **0.655** | [0.571, 0.743] |
| signature cumulative L=2 | 132 | 0.097 | 0.004 | -0.000 | **0.685** | [0.606, 0.759] |
| signature cumulative L=3 | 1463 | 0.058 | 0.000 | -0.000 | **0.666** | [0.590, 0.743] |
| signature raw L=2, no t | 110 | 0.051 | 0.237 | -0.001 | **0.632** | [0.537, 0.726] |
| signature raw L=3, no t | 1110 | 0.052 | 0.239 | -0.001 | **0.645** | [0.560, 0.733] |
| signature raw L=2 + length | 134 | 0.087 | 0.207 | -0.001 | **0.640** | [0.540, 0.738] |
| bag-of-symbols + signature raw L=2 | 346 | 0.118 | 0.212 | -0.001 | **0.624** | [0.525, 0.722] |

### retail  (n=1824, 114 tasks, pass rate 0.726)

| representation | p | ARI task | ARI model | ARI trial | AUC success | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| trajectory length | 2 | 0.017 | 0.012 | -0.001 | **0.517** | [0.456, 0.580] |
| bag-of-symbols | 214 | 0.398 | 0.000 | -0.001 | **0.585** | [0.522, 0.647] |
| TF-IDF symbol 1-3grams | 1282 | 0.294 | 0.343 | -0.001 | **0.654** | [0.604, 0.701] |
| LSA text (embedding stand-in) | 384 | 0.274 | 0.011 | -0.001 | **0.647** | [0.596, 0.693] |
| signature raw L=2 | 132 | 0.076 | 0.269 | -0.001 | **0.611** | [0.557, 0.658] |
| signature raw L=3 | 1463 | 0.052 | 0.262 | -0.001 | **0.611** | [0.564, 0.659] |
| signature cumulative L=2 | 132 | 0.096 | 0.006 | -0.001 | **0.590** | [0.540, 0.637] |
| signature cumulative L=3 | 1463 | 0.076 | 0.002 | -0.000 | **0.588** | [0.543, 0.633] |
| signature raw L=2, no t | 110 | 0.055 | 0.261 | -0.001 | **0.563** | [0.514, 0.612] |
| signature raw L=3, no t | 1110 | 0.047 | 0.245 | -0.001 | **0.580** | [0.531, 0.629] |
| signature raw L=2 + length | 134 | 0.081 | 0.270 | -0.001 | **0.610** | [0.561, 0.659] |
| bag-of-symbols + signature raw L=2 | 346 | 0.177 | 0.270 | -0.001 | **0.608** | [0.549, 0.667] |

### telecom  (n=1824, 114 tasks, pass rate 0.424)

| representation | p | ARI task | ARI model | ARI trial | AUC success | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| trajectory length | 2 | 0.010 | 0.041 | -0.000 | **0.729** | [0.690, 0.767] |
| bag-of-symbols | 214 | 0.090 | 0.004 | -0.001 | **0.891** | [0.855, 0.921] |
| TF-IDF symbol 1-3grams | 2444 | 0.067 | 0.007 | -0.001 | **0.877** | [0.846, 0.903] |
| LSA text (embedding stand-in) | 384 | 0.020 | 0.091 | -0.001 | **0.970** | [0.954, 0.981] |
| signature raw L=2 | 132 | 0.029 | 0.043 | -0.001 | **0.898** | [0.863, 0.927] |
| signature raw L=3 | 1463 | 0.025 | 0.044 | -0.000 | **0.904** | [0.878, 0.926] |
| signature cumulative L=2 | 132 | 0.030 | 0.030 | -0.000 | **0.843** | [0.809, 0.872] |
| signature cumulative L=3 | 1463 | 0.025 | 0.020 | -0.000 | **0.857** | [0.827, 0.885] |
| signature raw L=2, no t | 110 | 0.026 | 0.040 | -0.001 | **0.895** | [0.861, 0.923] |
| signature raw L=3, no t | 1110 | 0.028 | 0.040 | 0.000 | **0.895** | [0.865, 0.920] |
| signature raw L=2 + length | 134 | 0.027 | 0.040 | -0.000 | **0.899** | [0.864, 0.928] |
| bag-of-symbols + signature raw L=2 | 346 | 0.067 | 0.019 | -0.001 | **0.903** | [0.868, 0.931] |

### pooled  (n=4448, 278 tasks, pass rate 0.568)

| representation | p | ARI task | ARI model | ARI trial | AUC success | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| trajectory length | 2 | 0.014 | 0.016 | -0.000 | **0.689** | [0.650, 0.726] |
| bag-of-symbols | 214 | 0.201 | 0.000 | -0.000 | **0.759** | [0.721, 0.794] |
| TF-IDF symbol 1-3grams | 4318 | 0.155 | 0.000 | -0.000 | **0.787** | [0.755, 0.818] |
| LSA text (embedding stand-in) | 384 | 0.428 | -0.000 | -0.000 | **0.844** | [0.812, 0.874] |
| signature raw L=2 | 132 | 0.059 | 0.007 | -0.000 | **0.727** | [0.692, 0.763] |
| signature raw L=3 | 1463 | 0.055 | 0.006 | -0.000 | **0.784** | [0.754, 0.813] |
| signature cumulative L=2 | 132 | 0.052 | 0.007 | -0.000 | **0.724** | [0.690, 0.756] |
| signature cumulative L=3 | 1463 | 0.037 | 0.004 | -0.000 | **0.750** | [0.719, 0.781] |
| signature raw L=2, no t | 110 | 0.039 | 0.009 | -0.000 | **0.725** | [0.688, 0.762] |
| signature raw L=3, no t | 1110 | 0.038 | 0.006 | -0.000 | **0.778** | [0.747, 0.809] |
| signature raw L=2 + length | 134 | 0.060 | 0.007 | -0.000 | **0.729** | [0.693, 0.765] |
| bag-of-symbols + signature raw L=2 | 346 | 0.139 | 0.005 | -0.000 | **0.765** | [0.728, 0.801] |

### 4a. Reading the clusters

The `trial` control is 0.000 everywhere, as it must be.

**On the task-ID partition, plain counts win, everywhere, by a wide margin.**
Bag-of-symbols scores 0.257 / 0.398 / 0.090 / 0.201 (airline / retail / telecom
/ pooled) against 0.025-0.097 for every signature variant. That is a 3-7x gap,
and it does not close at depth 3, with the cumulative lift, or by concatenating
counts and signature.

**On the model-ID partition the ordering flips.** Signatures score 0.240
(airline) and 0.269 (retail) where bag-of-symbols scores 0.021 and 0.000. This
is the pattern the study was set up to detect, and the honest reading is the
unflattering one: *the signature separates who wrote the trace far better than
what the trace was trying to do.* On telecom, where traces are long and
dominated by user-simulator tool calls, even the model signal collapses to 0.043.

The `lsa_text` row at pooled scope (ARI task 0.428) is not a real win: pooling
mixes three domains with near-disjoint vocabularies, and any text
representation separates domains for free. That is exactly why the primary
analysis is within-domain.

### 4b. Predicting task success — the paired test

Overlapping marginal CIs are not a comparison, so every contrast below is a
**paired** task-level cluster bootstrap on the AUC *difference*, reusing the
same out-of-fold predictions (`compare.py`). `*` marks a 95% interval excluding
zero.

| contrast | airline | retail | telecom | pooled |
|---|---|---|---|---|
| **sig raw L2 - bag-of-symbols** | +0.015 (p=0.67) | +0.026 (p=0.39) | +0.007 (p=0.34) | **-0.032*** (p=0.013) |
| **sig raw L3 - bag-of-symbols** | +0.047 (p=0.19) | +0.026 (p=0.29) | +0.013 (p=0.14) | **+0.025*** (p=0.029) |
| sig cum L2 - bag-of-symbols | +0.078 (p=0.14) | +0.005 (p=0.89) | **-0.048*** (p<0.001) | **-0.036*** (p=0.004) |
| **TF-IDF k-grams - bag-of-symbols** | **+0.064*** (p=0.037) | **+0.069*** (p<0.001) | -0.015 (p=0.11) | **+0.028*** (p<0.001) |
| bag + sig raw L2 - bag alone | +0.017 (p=0.30) | +0.023 (p=0.05) | **+0.012*** (p=0.003) | **+0.005*** (p=0.047) |
| sig raw L2 - length | -0.013 (p=0.81) | **+0.094*** (p=0.011) | **+0.169*** (p<0.001) | **+0.039*** (p=0.004) |
| bag-of-symbols - length | -0.029 (p=0.65) | +0.068 (p=0.15) | **+0.162*** (p<0.001) | **+0.070*** (p<0.001) |

**The signature does not beat bag-of-symbols.** In all three individual domains
the depth-2 difference is indistinguishable from zero, and pooled it is
significantly *negative*. Depth 3 buys a significant +0.025 pooled — for 1463
features against 214, a poor trade. Adding the signature on top of counts moves
AUC by +0.005 to +0.023; the two significant gains are the two smallest.

**But order is not worthless here, and that is the sharpest result in the
study.** TF-IDF over symbol 1-3-grams beats bag-of-symbols significantly in
airline (+0.064), retail (+0.069) and pooled (+0.028). So ordered information
*does* carry predictive signal that counts miss — and a bag of k-grams extracts
it while the signature does not.

### 4c. The order-shuffle control

Bag-of-symbols is invariant to shuffling by construction; the signature is not.
Shuffling steps within each trajectory and recomputing (3 seeds, depth 2):

| scope | signature raw L2, intact | shuffled | drop |
|---|---:|---:|---:|
| airline | 0.622 | 0.577 +/- 0.021 | 0.045 |
| retail | 0.611 | 0.539 +/- 0.010 | 0.072 |
| telecom | 0.898 | 0.681 +/- 0.007 | **0.217** |
| pooled | 0.727 | 0.680 +/- 0.008 | 0.047 |

So the signature genuinely reads order — on telecom, order is worth 0.22 AUC to
it. Put that next to §4b and the picture resolves: an 11-channel *ordered*
summary reaches parity with a 214-dimensional *unordered* one. Order buys back
exactly what the channel compression threw away, and nothing beyond it.

### 4d. The anti-cancellation guard is inert on this corpus

The monotone index is theoretically motivated: a tree-like path has trivial
signature. It does not bite here (`treelike.py`, all 4448 traces):

- exact immediate backtracks are **0.243%** of 201 273 path segments;
- **zero** trajectories have a degenerate signature without `t`, at any
  threshold down to 1e-2 (minimum `||sig||` without `t` is 0.376);
- signature norms with and without `t` correlate at **0.9995**.

Empirically the guard changes AUC significantly in exactly one of four scopes
(retail, +0.048 at L2, p=0.001) and is null in the other three (airline -0.010
p=0.62; telecom +0.003 p=0.22; pooled +0.003 p=0.52). Keep it — it is nearly
free and the failure mode it prevents is real — but it is not what is holding
this representation back.

### 4e. A caveat on the text baseline, and on telecom

`lsa_text` scores 0.970 on telecom. That is not a language model understanding
agent behaviour; it is reading the outcome off the transcript. The bare token
`transfer` appears in 1425 of 1824 telecom traces and splits pass rate by
**-0.680**. The symbol `call:a:transfer_to_human_agents:a3` splits it by
-0.569. Telecom "success prediction" is largely *did the agent give up*, which
any representation containing that symbol answers.

Note the sign is domain-specific: in airline, transfer correlates *positively*
with reward (+0.294), because some airline tasks are supposed to end in a
transfer. So this is not one universal give-up detector.

Two honest consequences. First, the canonicalisation deliberately strips free
text, which is right for testing *structure* but handicaps it against a text
model on a task that is partly a text-reading task. Second, telecom's high
absolute AUCs (0.84-0.97) should not be read as "structure works well here".

---

## 5. Interpretability

`interpret.py` decodes the standardised logistic-regression weights back to
signature level and channel. (Caveat: this fit is on all rows at the CV-selected
`C`, so it describes in-sample structure of a classifier whose honest AUC is
0.61-0.90.)

**Almost all the weight is at level 2, and on the raw path it is entirely Lévy
area.** Level-1 mass is 1.6% (airline) and 1.7% (retail) of total |weight|; the
level-2 weights come in exactly antisymmetric pairs (`is_write -> log_out_size`
at -0.191 against `log_out_size -> is_write` at +0.191, and so on down the
list). The classifier is using signed areas and nothing else.

**Why**, and it is the mechanism behind the whole result: level-1 terms are
exactly `X[-1] - X[0]`, and the `R` columns are bounded indicators, so on the
raw path they are pinned near zero.

| | mean abs level-1 term, raw path | cumulative path |
|---|---:|---:|
| symbol-block channels (8) | **0.023** | 8.500 |
| `is_error` | 0.0000 (always exactly 0) | 0.284 |
| `target_novel` | 0.0000 (always exactly 0) | 7.634 |

**76.1% of raw-path level-1 coordinates are exactly zero.** The literal
`E = R[s]` lift throws away all count-like information by construction, which is
precisely the information bag-of-symbols is made of. That is not a subtle
modelling choice — it is the difference between the two lifts, and it is why
`sig_cum` behaves like counts (model ARI 0.004-0.006) while `sig_raw` behaves
like a style detector (0.24-0.27).

**Do the surviving coordinates map to human-readable statements? Some do,
exactly.** Because `t_index` runs 0 -> 1 exactly once, `S^{t,j} = INT t dj`, so

> `S^{t,j} / (total movement of j)` **is** the increment-weighted mean position
> at which channel `j` moves.

Verified numerically: correlation 1.000000 against a directly computed weighted
mean position (`out/levy_check.txt`). For a channel that only increases, this
reads directly as *"how late in the trajectory does j accumulate"*. So the
retail model's largest weight, `(t_index, cum_err_rate)` at -0.575, is a
genuine behavioural statement. Both channels are monotone, so the reading is
exact: a larger signed area means errors accumulate *later*, and the negative
coefficient makes that predict failure — **errors arriving late in a trajectory
predict failure; the same errors early do not.**

That is the only pair in the top-10 that can be read this cleanly, because
`t_index` and `cum_err_rate` are the only two monotone channels. The next
largest, `(is_search, cum_err_rate)` at +0.422, nominally says search activity
leading error accumulation predicts success — but `is_search` moves up and back
down on the raw path, so its "mean position" is a signed average without a
plain-language meaning, and the sentence should not be trusted.

**Where it stops being readable.** For channels whose increments change sign on
the raw path (`log_out_size`, and every 0/1 indicator, which goes up and back
down), the same coordinate is a signed weighted average that can fall outside
[0, 1] and has no plain-language reading. The telecom model's top coordinate is
`log_out_size -> log_out_size`, a level-2 *diagonal* — which is just
`(total output size)^2 / 2`, i.e. trace bulk in disguise, not an interaction at
all. And there are 132 coordinates at depth 2 and 1463 at depth 3: a handful
decode into sentences, the rest do not.

So: **partially interpretable, and more than a random projection would be** —
the pairs against the monotone index have an exact reading, which is a real
argument for the hand-designed `R` over a Gaussian one. But most coordinates do
not map to any statement about agent behaviour.

### 5a. Is the model-ID signal behaviour or verbosity?

The alphabet buckets natural-language turn length, which is about the most
model-characteristic and least behavioural thing available. Collapsing those
buckets (`say:agent:l2 -> say:agent`, V 214 -> 208) and re-measuring:

| scope | representation | ARI model, with verbosity | without | ARI task, with | without |
|---|---|---:|---:|---:|---:|
| airline | sig raw L2 | 0.240 | **0.135** | 0.079 | 0.076 |
| airline | TF-IDF k-grams | 0.265 | **0.083** | 0.200 | 0.218 |
| retail | sig raw L2 | 0.269 | **0.236** | 0.076 | 0.070 |
| retail | TF-IDF k-grams | 0.343 | **0.225** | 0.294 | 0.265 |
| telecom | sig raw L2 | 0.043 | 0.074 | 0.029 | 0.025 |

Verbosity accounts for roughly **44%** of the signature's model signal in
airline and **12%** in retail (and 69% / 34% of the k-gram baseline's). So the
answer is *partly*: turn length is a large chunk of it in airline, but a
substantial model fingerprint survives in retail that is not turn length —
presumably tool-mix, argument-size and retry habits, which are behavioural.
Either way the task-ID ARI is untouched, so this does not rescue the
representation on the partition that matters.

---

## 6. Verdict

**Does cluster structure exist?** Yes, on both partitions, and the negative
control (trial index, ARI 0.000) confirms the measurement is not manufacturing
it. Task structure is strong in retail (bag-of-symbols ARI 0.398), moderate in
airline (0.257), weak in telecom (0.090). Model structure exists at 0.24-0.27 in
airline and retail.

**Does the signature find it? No.** On task ID the signature reaches 0.025-0.097
against counts' 0.090-0.398 — beaten 3-7x in every domain. The only partition it
reads better than counts is model ID, and about half of that (airline) is
response verbosity. This is the "reading style, not behaviour" outcome, and it
is what the data says.

**Does the ordered/global representation beat bag-of-symbols? No.** Paired
bootstrap on AUC differences: +0.015 (p=0.67) airline, +0.026 (p=0.39) retail,
+0.007 (p=0.34) telecom, and **-0.032 (p=0.013)** pooled. Depth 3 gets a
significant +0.025 pooled for 7x the dimension. Stacked on top of counts the
signature adds +0.005 to +0.023.

**This is not because order carries no signal.** Two results say it does:
shuffling steps costs the signature 0.045-0.217 AUC, and TF-IDF over symbol
k-grams beats bag-of-symbols by a significant +0.064 / +0.069 / +0.028 in
airline / retail / pooled. **Order matters and the signature is the wrong
extractor for it here.** Three concrete reasons, all measured:

1. **The raw lift deletes the counts.** 76% of level-1 coordinates are exactly
   zero because bounded indicator channels return to where they started; the
   signature spends its budget on areas and never sees the base rates that carry
   most of the task signal.
2. **Reparametrisation invariance discards trajectory length**, which is the
   strongest scalar in the corpus (point-biserial -0.34 with reward, and length
   alone reaches AUC 0.729 on telecom). Bolting length back on moves AUC by
   -0.001 to +0.017 (largest in airline, null elsewhere) — so the signature had
   largely captured it by other means, but it started from a self-imposed
   handicap it did not need.
3. **The channel bottleneck.** Compressing 214 symbols into 11 hand-designed
   channels loses more than the ordering recovers. The shuffle control quantifies
   the trade exactly: an ordered 11-channel view lands at parity with an
   unordered 214-dimensional one.

**Is any of it interpretable? Partly, and this is the one place the approach
earns its keep.** The hand-designed `R` plus the monotone index gives an exact
reading for one family of coordinates: `S^{t,j}/dj` *is* the mean position at
which channel `j` moves (correlation 1.000000, verified). That turns the retail
model's top weight into a real sentence — errors late in a trajectory predict
failure, errors early do not — which a random Gaussian `R` could never have
produced. But this covers a small minority of the 132 (or 1463) coordinates;
most decode to nothing sayable, and the single largest telecom weight is a
level-2 diagonal that is just total output size squared.

**Bottom line.** On short, tool-grounded agent traces with binary reward, the
path signature is a *negative result as a behavioural representation*: it does
not recover task structure, it does recover model identity (partly as verbosity),
and it does not beat a bag-of-symbols count vector at predicting success in any
domain tested. If the goal is order-sensitive prediction on this kind of data,
symbol k-grams are cheaper, stronger and already significant. The parts worth
keeping from this exercise are the hand-designed interpretable channel table and
the exact timing reading of the level-2 coordinates against the monotone index —
not the signature transform itself.

### What would change the verdict

- **Lift with `cumsum` and take areas on top of counts**, rather than choosing
  between them. `sig_cum` level-1 *is* bag-of-symbols; the level-2 block on that
  path is the honest "counts plus ordering" object. It was not tested here as a
  residual-over-counts model, and `bag + sig_raw` (+0.005 to +0.023) is a weak
  proxy for it.
- **Longer, more branching trajectories.** Median 28-65 steps with 0.24%
  backtracking is a regime where the signature's distinguishing machinery —
  cancellation, area, iterated structure — has little to work with. SWE-bench-style
  traces with real backtracking would be a fairer test, and the tree-likeness
  diagnostic in `treelike.py` is the right thing to check first.
- **More channels, or channels chosen for variance.** `is_error` and
  `target_novel` contribute a level-1 term of exactly zero on every trace in the
  corpus. That is a design bug in the lift, not a fact about signatures.

---

## Limitations

- **No pretrained sentence embeddings.** This environment's network policy
  blocks `huggingface.co` (and every model host tried), so the "sentence
  embeddings of flattened trace" baseline is substituted with TF-IDF word
  1-2-grams -> TruncatedSVD(384), labelled `lsa_text` throughout. A real
  sentence encoder would likely score *higher*, which would strengthen, not
  weaken, the negative verdict on signatures.
- **TF-IDF and SVD are fit on each scope's full row set** (unsupervised, no
  labels), i.e. transductively. This can only help the text and k-gram
  baselines, which are the ones already beating the signature.
- **Interpretability weights are in-sample** at the CV-selected `C`; they
  describe the fitted classifier, not an out-of-fold effect.
- **One benchmark family.** tau2-bench is customer-service dialogue with tool
  use. The verdict is about this regime, not about path signatures generally.
- KMeans is given the true `k`, and ARI at `k`=114 with 1824 points is a
  demanding measure — but it is applied identically to every representation, so
  the comparisons hold even where absolute values are low.
