"""Task sampler and the zero-signal rule. Pure Python, deterministic given its state.

State (a JSON dict, snapshotted with every eval point):
  run, name     seed the per-pool, per-epoch permutations: sorted pool shuffled by sha256("run|name|pool|epoch")
  rollouts      lineage rollouts trained so far (the clock rests are measured on)
  pools         per pool: epoch and cursor into that epoch's permutation
  streak        task -> zero-signal visits in a row (only nonzero entries)
  rest_until    task -> lineage rollout count at which it returns

assign() gives the rows of the next generations (P prompts each): round(P * hard_share) from the hard pool, the rest
from the easy pool, skipping rested tasks; a pool with no active task is filled from the other.
update() applies the rule after a segment: a visit is zero-signal when all G rollouts of the prompt got the same
training reward; ZERO_STREAK such visits in a row rest the task for REST_ROLLOUTS lineage rollouts; a visit with
signal resets the streak; a task returns with streak 0. The rule acts at segment boundaries.
The orchestrator runs both; the same code and rule serve the overseer arm and A2.

  python3 sampler.py   self-check
"""
import copy
import hashlib
import random

GEN_ROLLOUTS = 32  # as knobs.GEN_ROLLOUTS (asserted in the self-check)
ZERO_STREAK = 2  # [default]
REST_ROLLOUTS = 3200  # [default]
POOLS = ("easy", "hard")


def new(run, name):
    return {"run": run, "name": name, "rollouts": 0, "pools": {p: {"epoch": 0, "cursor": 0} for p in POOLS},
            "streak": {}, "rest_until": {}}


def _perm(state, name, pool, epoch):
    p = sorted(pool)
    random.Random(hashlib.sha256(f"{state['run']}|{state['name']}|{name}|{epoch}".encode()).digest()).shuffle(p)
    return p


def active(state, pool, at=None):
    at = state["rollouts"] if at is None else at
    return [t for t in pool if state["rest_until"].get(t, 0) <= at]


def _take(state, name, pool, n, at):
    """The next n active tasks of one pool at lineage rollout count `at`; [] if the pool has no active task."""
    out = []
    if n <= 0 or not active(state, pool, at):
        return out
    ps = state["pools"][name]
    order = _perm(state, name, pool, ps["epoch"])
    while len(out) < n:
        if ps["cursor"] >= len(order):
            ps["epoch"], ps["cursor"] = ps["epoch"] + 1, 0
            order = _perm(state, name, pool, ps["epoch"])
        t = order[ps["cursor"]]
        ps["cursor"] += 1
        if state["rest_until"].get(t, 0) <= at:
            out.append(t)
    return out


def assign(state, pools, P, hard_share, gens):
    """Task ids for the next `gens` generations, P per generation. Advances the cursors; update() advances the clock.
    Known limit: one task can appear twice in a generation when an epoch wraps inside it (or a pool has under P active
    tasks); readers group rollouts by prompt index, not task id, so it is harmless. Upgrade if pools get tiny."""
    rows = []
    for i in range(gens):
        at = state["rollouts"] + i * GEN_ROLLOUTS
        hard = _take(state, "hard", pools["hard"], round(P * hard_share), at)
        easy = _take(state, "easy", pools["easy"], P - len(hard), at)
        if len(hard) + len(easy) < P:  # no active easy task: the rest comes from the hard pool
            hard += _take(state, "hard", pools["hard"], P - len(hard) - len(easy), at)
        assert len(hard) + len(easy) == P, "no active task in either pool"
        rows += hard + easy
    return rows


def update(state, visits, gens):
    """After a segment of `gens` generations: visits = [(task_id, [the training rewards of its G rollouts])] in order."""
    state["rollouts"] += gens * GEN_ROLLOUTS
    now, rested = state["rollouts"], set()
    for t, rewards in visits:
        if t in rested:  # rested earlier in this segment: later visits in it do not count
            continue
        if len(set(rewards)) > 1:
            state["streak"].pop(t, None)
            continue
        k = state["streak"].pop(t, 0) + 1
        if k >= ZERO_STREAK:
            state["rest_until"][t] = now + REST_ROLLOUTS
            rested.add(t)
        else:
            state["streak"][t] = k
    for t in [t for t, u in state["rest_until"].items() if u <= now]:  # returned: streak is already 0
        del state["rest_until"][t]


def counts(state, pools):
    """Active and rested task counts per pool at the state's clock."""
    return {p: {"active": len(active(state, pools[p])), "rested": len(pools[p]) - len(active(state, pools[p]))} for p in POOLS}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import knobs

    assert GEN_ROLLOUTS == knobs.GEN_ROLLOUTS
    pools = {"easy": [f"e{i:03d}" for i in range(40)], "hard": [f"h{i:03d}" for i in range(12)]}
    SIG, ZERO = [0.0, 1.0, 0.0, 0.0], [0.0] * 4

    def seg(s, P, h, gens, rew):
        rows = assign(s, pools, P, h, gens)
        update(s, [(t, rew(t)) for t in rows], gens)
        return rows

    # determinism, and independence from pool order and from other streams
    a, b = new("r", "root"), new("r", "root")
    ra = seg(a, 8, 0.25, 5, lambda t: SIG)
    assert ra == assign(b, {k: list(reversed(v)) for k, v in pools.items()}, 8, 0.25, 5) and len(ra) == 40
    assert ra != assign(new("r", "a2s2"), pools, 8, 0.25, 5) and ra != assign(new("r2", "root"), pools, 8, 0.25, 5)
    # P = 8 and P = 4 with every share: round(P * h) hard prompts per generation, hard first
    for P in (8, 4):
        for h in (0.0, 0.25, 0.5):
            rows = assign(new("r", "x"), pools, P, h, 3)
            for g in range(3):
                gen = rows[g * P:(g + 1) * P]
                assert len(gen) == P and sum(t[0] == "h" for t in gen) == round(P * h) == sum(t[0] == "h" for t in gen[:round(P * h)])
    # epoch coverage: every active task once per epoch, and epochs differ
    s = new("r", "root")
    rows = assign(s, pools, 8, 0.0, 10)  # 80 rows = two epochs of 40
    assert sorted(rows[:40]) == sorted(rows[40:]) == sorted(pools["easy"]) and rows[:40] != rows[40:]
    # snapshot and restore give the same continuation; the abandoned branch leaves no trace
    s = new("r", "root")
    seg(s, 8, 0.25, 4, lambda t: ZERO if t.endswith("3") else SIG)
    snap = copy.deepcopy(s)
    cont = [seg(s, 4, 0.5, 6, lambda t: ZERO if t.endswith("3") else SIG) for _ in range(3)]
    r = copy.deepcopy(snap)
    assert [seg(r, 4, 0.5, 6, lambda t: ZERO if t.endswith("3") else SIG) for _ in range(3)] == cont and r == s
    r2 = copy.deepcopy(snap)
    assert seg(r2, 4, 0.5, 6, lambda t: SIG) == cont[0] and r2 != s  # same rows, different outcomes, different state
    # rest and return timing: two zero-signal visits in a row rest a task for REST_ROLLOUTS lineage rollouts
    s = new("r", "root")
    one = {"easy": ["a", "b", "c", "d"], "hard": []}
    assert sorted(assign(s, one, 4, 0.0, 1)) == ["a", "b", "c", "d"]
    update(s, [("a", ZERO), ("b", SIG), ("c", ZERO), ("d", [0.25] * 4)], 1)
    assert s["streak"] == {"a": 1, "c": 1, "d": 1} and not s["rest_until"] and s["rollouts"] == 32
    update(s, [("a", ZERO), ("b", ZERO), ("c", SIG), ("d", [0.25, 0.25, 0.25, 0.0])], 1)  # a: 2 in a row; c and d: reset
    assert s["streak"] == {"b": 1} and s["rest_until"] == {"a": 64 + REST_ROLLOUTS} and s["rollouts"] == 64
    assert counts(s, one) == {"easy": {"active": 3, "rested": 1}, "hard": {"active": 0, "rested": 0}}
    gens_rest = REST_ROLLOUTS // GEN_ROLLOUTS
    rows = assign(s, one, 4, 0.0, gens_rest + 2)
    resting, back = rows[:4 * gens_rest], rows[4 * gens_rest:]
    assert "a" not in resting and "a" in back, "rested task sampled early or not returned on time"
    update(s, [], gens_rest - 1)
    assert "a" in s["rest_until"]
    update(s, [], 1)
    assert "a" not in s["rest_until"] and "a" not in s["streak"]  # back, streak 0
    update(s, [("a", ZERO)], 1)
    assert s["streak"]["a"] == 1 and "a" not in s["rest_until"]  # one visit after the return does not rest it again
    # three zero visits inside one segment: rested once, the third does not count
    s = new("r", "root")
    update(s, [("a", ZERO), ("a", ZERO), ("a", ZERO)], 2)
    assert s["rest_until"] == {"a": 64 + REST_ROLLOUTS} and not s["streak"]
    # empty-pool fill, both ways; nothing active at all is an error
    s = new("r", "root")
    assert all(t[0] == "e" for t in assign(s, {"easy": pools["easy"], "hard": []}, 8, 0.5, 2))
    s["rest_until"] = {t: 10 ** 9 for t in pools["easy"]}
    assert all(t[0] == "h" for t in assign(s, pools, 8, 0.25, 2))
    s["rest_until"].update({t: 10 ** 9 for t in pools["hard"]})
    try:
        assign(s, pools, 8, 0.25, 1)
        raise SystemExit("assign with no active task did not fail")
    except AssertionError:
        pass
    # a rested task is skipped inside an epoch and the rest of the epoch still covers every active task once
    s = new("r", "root")
    s["rest_until"] = {"e000": 10 ** 9, "e001": 10 ** 9}
    rows = assign(s, pools, 8, 0.0, 5)[:38]
    assert sorted(rows) == sorted(set(pools["easy"]) - {"e000", "e001"})
    print("SAMPLER_CHECK ok", "ZERO_STREAK", ZERO_STREAK, "REST_ROLLOUTS", REST_ROLLOUTS)
