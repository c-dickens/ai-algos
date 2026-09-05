"""Stage 1a: stream tau2-bench leaderboard result files into compact per-simulation
step records. Drops all natural-language payloads except their length, keeps a
separate raw-text dump for the text baselines.

Output: out/steps.jsonl (one simulation per line), out/text.jsonl (flattened text).
"""
import json, os, sys, glob, collections

RESULTS = "/home/user/sierra-research/tau2-bench/data/tau2/results/final"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Only plain `llm_agent` runs on the three standard domains. The telecom-workflow
# domain and the _solo / _gt ablations change the scaffold, not the model, so
# including them would confound the model-ID partition.
KEEP = [
    ("claude-3-7-sonnet-20250219", "airline"), ("claude-3-7-sonnet-20250219", "retail"),
    ("claude-3-7-sonnet-20250219", "telecom"),
    ("gpt-4.1-2025-04-14", "airline"), ("gpt-4.1-2025-04-14", "retail"),
    ("gpt-4.1-2025-04-14", "telecom"),
    ("gpt-4.1-mini-2025-04-14", "airline"), ("gpt-4.1-mini-2025-04-14", "retail"),
    ("gpt-4.1-mini-2025-04-14", "telecom"),
    ("o4-mini-2025-04-16", "airline"), ("o4-mini-2025-04-16", "retail"),
    ("o4-mini-2025-04-16", "telecom"),
]
FILES = {}
for model, domain in KEEP:
    for cfg in ("default", "base"):
        p = f"{RESULTS}/{model}_{domain}_{cfg}_gpt-4.1-2025-04-14_4trials.json"
        if os.path.exists(p):
            FILES[(model, domain)] = p
assert len(FILES) == 12, sorted(FILES)


def unroll(messages):
    """messages -> flat list of step dicts. One step per tool call, per tool
    result, and per natural-language turn."""
    call_meta = {}          # tool_call_id -> (name, requestor)
    steps = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for j, tc in enumerate(tcs):
                    call_meta[tc["id"]] = (tc["name"], tc.get("requestor", "assistant"))
                    args = tc.get("arguments") or {}
                    steps.append(dict(
                        kind="call", tool=tc["name"],
                        requestor=tc.get("requestor", "assistant"),
                        n_args=len(args), args_chars=len(json.dumps(args, default=str)),
                        # position inside this parallel tool-call batch, and its size
                        batch_idx=j, batch_size=len(tcs),
                        # canonical target: the argument values, ID-stripped later
                        target=json.dumps(sorted(map(str, args.values())), default=str),
                        out_chars=0, error=False,
                    ))
            if m.get("content"):
                steps.append(dict(kind="say", tool="agent", requestor="assistant",
                                  n_args=0, args_chars=0, batch_idx=0, batch_size=1,
                                  target="", out_chars=len(m["content"]), error=False))
        elif role == "user":
            tcs = m.get("tool_calls") or []
            for j, tc in enumerate(tcs):
                call_meta[tc["id"]] = (tc["name"], tc.get("requestor", "user"))
                args = tc.get("arguments") or {}
                steps.append(dict(kind="call", tool=tc["name"], requestor="user",
                                  n_args=len(args), args_chars=len(json.dumps(args, default=str)),
                                  batch_idx=j, batch_size=len(tcs),
                                  target=json.dumps(sorted(map(str, args.values())), default=str),
                                  out_chars=0, error=False))
            if m.get("content"):
                steps.append(dict(kind="say", tool="user", requestor="user",
                                  n_args=0, args_chars=0, batch_idx=0, batch_size=1,
                                  target="", out_chars=len(m["content"]), error=False))
        elif role == "tool":
            name, req = call_meta.get(m.get("id"), ("<unknown>", m.get("requestor", "assistant")))
            content = m.get("content") or ""
            steps.append(dict(kind="res", tool=name, requestor=req,
                              n_args=0, args_chars=0, batch_idx=0, batch_size=1,
                              target="", out_chars=len(content),
                              error=bool(m.get("error", False))))
    return steps


def flat_text(messages):
    parts = []
    for m in messages:
        role = m.get("role")
        if m.get("content"):
            parts.append(f"{role}: {m['content']}")
        for tc in (m.get("tool_calls") or []):
            parts.append(f"{role} calls {tc['name']}({json.dumps(tc.get('arguments') or {}, default=str)})")
    return "\n".join(parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = open(os.path.join(OUT, "steps.jsonl"), "w")
    ft = open(os.path.join(OUT, "text.jsonl"), "w")
    counts = collections.Counter()
    for (model, domain), path in sorted(FILES.items()):
        d = json.load(open(path))
        for sim in d["simulations"]:
            rec = dict(
                sim_id=sim["id"], model=model, domain=domain,
                task_id=f"{domain}/{sim['task_id']}", trial=sim["trial"],
                reward=float(sim["reward_info"]["reward"]),
                termination_reason=sim.get("termination_reason"),
                n_messages=len(sim["messages"]),
                steps=unroll(sim["messages"]),
            )
            fs.write(json.dumps(rec) + "\n")
            ft.write(json.dumps(dict(sim_id=sim["id"], text=flat_text(sim["messages"]))) + "\n")
            counts[(model, domain)] += 1
        del d
        print(f"  {model:28s} {domain:8s} {counts[(model, domain)]:5d} sims", flush=True)
    fs.close(); ft.close()
    print("total simulations:", sum(counts.values()))


if __name__ == "__main__":
    main()
