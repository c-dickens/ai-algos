"""Stage 1b: step -> symbol string -> interned integer, plus the hand-designed
lookup table R (V x 8) and the per-step contextual channels.

A note on the V x 8 table. Four of the eight requested columns (is_read,
is_write, is_search, is_error) are intrinsic to a symbol, and arg_size_bucket is
too *because we put the bucket into the symbol string*. The remaining three
(retry_flag, target_novel, depth) are history-dependent: whether a call is a
retry cannot be a function of the symbol alone. So R is a genuine V x 5 lookup
and three columns are computed per step; together they still form the 8-column
symbol block the design asks for.
"""
import json, math, numpy as np

# ---------------------------------------------------------------- log buckets
ARG_EDGES = (8, 32, 128)      # chars of serialised arguments
SAY_EDGES = (64, 192, 512)    # chars of natural-language content


def bucket(x, edges):
    b = 0
    for e in edges:
        if x >= e:
            b += 1
    return b


# ------------------------------------------------------- hand-coded semantics
READ_PREFIX = ("get_", "check_", "list_", "find_", "can_", "run_speed_test")
WRITE_PREFIX = ("book_", "cancel_", "update_", "modify_", "send_", "make_",
                "exchange_", "return_", "enable_", "disable_", "resume_",
                "suspend_", "refuel_", "reboot_", "reseat_", "reset_", "set_",
                "toggle_", "grant_", "disconnect_", "transfer_to_")
# "search" here means lookup-by-attribute: resolving an entity you do not have
# an ID for, as opposed to reading an entity you already identified.
SEARCH_EXACT = ("search_direct_flight", "search_onestop_flight",
                "find_user_id_by_email", "find_user_id_by_name_zip",
                "get_customer_by_name", "get_customer_by_phone",
                "list_all_airports", "list_all_product_types")


def tool_semantics(tool):
    is_read = float(tool.startswith(READ_PREFIX))
    is_write = float(tool.startswith(WRITE_PREFIX))
    is_search = float(tool in SEARCH_EXACT)
    return is_read, is_write, is_search


# ------------------------------------------------------------------- symbols
def symbol(step):
    """Canonical symbol for one step. No IDs, no timestamps, no free text;
    continuous quantities appear only as log-scale bucket indices."""
    req = "a" if step["requestor"] == "assistant" else "u"
    if step["kind"] == "call":
        return f"call:{req}:{step['tool']}:a{bucket(step['args_chars'], ARG_EDGES)}"
    if step["kind"] == "res":
        return f"res:{req}:{step['tool']}:{'err' if step['error'] else 'ok'}"
    return f"say:{step['tool']}:l{bucket(step['out_chars'], SAY_EDGES)}"


def build_vocab(records):
    counts = {}
    for r in records:
        for s in r["steps"]:
            sym = symbol(s)
            counts[sym] = counts.get(sym, 0) + 1
    vocab = {s: i for i, s in enumerate(sorted(counts))}
    return vocab, counts


R_COLS = ["is_read", "is_write", "is_search", "is_error", "arg_size_bucket"]
CTX_COLS = ["retry_flag", "target_novel", "depth"]
EXTRA_COLS = ["t_index", "cum_err_rate", "log_out_size"]
CHANNELS = R_COLS + CTX_COLS + EXTRA_COLS          # d = 11


def build_R(vocab):
    """V x 5 interpretable lookup. Columns are hand-assigned from tool
    semantics, not sampled."""
    R = np.zeros((len(vocab), len(R_COLS)), dtype=np.float64)
    for sym, i in vocab.items():
        parts = sym.split(":")
        if parts[0] == "call":
            tool = parts[2]
            rd, wr, se = tool_semantics(tool)
            R[i, 0], R[i, 1], R[i, 2] = rd, wr, se
            R[i, 3] = 0.0
            R[i, 4] = int(parts[3][1:]) / (len(ARG_EDGES))       # 0..1
        elif parts[0] == "res":
            tool = parts[2]
            rd, wr, se = tool_semantics(tool)
            R[i, 0], R[i, 1], R[i, 2] = rd, wr, se
            R[i, 3] = 1.0 if parts[3] == "err" else 0.0
            R[i, 4] = 0.0
        else:                                                     # say
            R[i, 4] = int(parts[2][1:]) / (len(SAY_EDGES))
    return R


def path(record, vocab, R, with_t=True, cumulative=False):
    """Build the (n, d) stream for one trajectory."""
    steps = record["steps"]
    n = len(steps)
    syms = [vocab[symbol(s)] for s in steps]
    X = np.zeros((n, len(CHANNELS)), dtype=np.float64)
    X[:, :len(R_COLS)] = R[syms]

    seen_tool_target, seen_target = set(), set()
    chain, cum_err = 0, 0
    for i, s in enumerate(steps):
        if s["kind"] == "call":
            key, tgt = (s["tool"], s["target"]), s["target"]
            X[i, 5] = 1.0 if key in seen_tool_target else 0.0      # retry_flag
            X[i, 6] = 0.0 if tgt in seen_target else 1.0           # target_novel
            seen_tool_target.add(key); seen_target.add(tgt)
        if s["kind"] == "say":
            chain = 0
        else:
            chain += 1
        X[i, 7] = min(math.log1p(chain) / math.log(20.0), 1.5)     # depth
        if s["kind"] == "res" and s["error"]:
            cum_err += 1
        X[i, 9] = cum_err / n                                      # cum err rate
        X[i, 10] = math.log1p(s["out_chars"]) / 10.0               # log out size
    X[:, 8] = np.arange(n) / max(n - 1, 1) if with_t else 0.0      # monotone index

    if cumulative:
        # channels 0..7 and 10 are per-step readings; 8 and 9 are already
        # monotone. Accumulating turns level-1 signature terms into plain counts.
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 10]
        X[:, idx] = np.cumsum(X[:, idx], axis=0)
    if not with_t:
        return np.delete(X, 8, axis=1)
    return X
