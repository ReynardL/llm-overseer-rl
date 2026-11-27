"""Tagged JSON for Python values, and the outside-the-sandbox comparison.

Used in two places: inside the sandbox (grade.py encodes the agent's outputs) and outside it (the grader decodes and
compares). Outside, only json.loads touches sandbox output, so nothing the agent wrote is ever executed there.
"""
import json
import math

MAX_DEPTH = 60
MAX_BYTES = 100_000  # per encoded value; bigger becomes ["?", "big"], which never matches


def encode(v, depth=0):
    if depth > MAX_DEPTH:
        return ["?", "depth"]
    t = type(v)
    if v is None:
        return ["n"]
    if t is bool:
        return ["b", v]
    if t is int:
        return ["i", v] if abs(v) < 10**4000 else ["?", "bigint"]
    if t is float:
        return ["f", repr(v)]
    if t is complex:
        return ["c", repr(v.real), repr(v.imag)]
    if t is str:
        return ["s", v]
    if t is bytes:
        return ["y", v.hex()]
    if t is list or t is tuple:
        return ["l" if t is list else "t", [encode(x, depth + 1) for x in v]]
    if isinstance(v, dict):  # dict, Counter, OrderedDict, defaultdict: compared as plain mappings
        items = [[encode(k, depth + 1), encode(x, depth + 1)] for k, x in v.items()]
        return ["d", sorted(items, key=lambda kv: json.dumps(kv[0]))]
    if t is set or t is frozenset:
        return ["S" if t is set else "F", sorted((encode(x, depth + 1) for x in v), key=json.dumps)]
    if t is range:
        return ["r", v.start, v.stop, v.step]
    return ["?", t.__name__]


def dumps(v):
    s = json.dumps(encode(v))
    return s if len(s) <= MAX_BYTES else json.dumps(["?", "big"])


def decode(e):
    tag = e[0]
    if tag == "n":
        return None
    if tag in ("b", "i", "s"):
        return e[1]
    if tag == "f":
        return float(e[1])
    if tag == "c":
        return complex(float(e[1]), float(e[2]))
    if tag == "y":
        return bytes.fromhex(e[1])
    if tag == "l":
        return [decode(x) for x in e[1]]
    if tag == "t":
        return tuple(decode(x) for x in e[1])
    if tag == "d":
        return {decode(k): decode(x) for k, x in e[1]}
    if tag == "S":
        return {decode(x) for x in e[1]}
    if tag == "F":
        return frozenset(decode(x) for x in e[1])
    if tag == "r":
        return range(e[1], e[2], e[3])
    raise ValueError(f"cannot decode {tag}")


def _num(e):
    return float(e[1]) if e[0] == "f" else e[1]


def same(a, b):
    """True if agent value a matches reference value b. Numbers compare like Python == with float tolerance;
    everything else structurally. Unknown types, timeouts and malformed values never match."""
    try:
        ta, tb = a[0], b[0]
        if ta in ("i", "f") and tb in ("i", "f"):
            x, y = _num(a), _num(b)
            if ta == tb == "i":
                return x == y
            if math.isnan(x) or math.isnan(y):
                return math.isnan(x) and math.isnan(y)
            return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)
        if ta != tb or ta in ("?", "T"):
            return False
        if ta in ("n",):
            return True
        if ta in ("b", "s", "y", "E"):
            return a[1] == b[1]
        if ta == "c":
            return all(same(["f", p], ["f", q]) for p, q in zip(a[1:], b[1:]))
        if ta == "r":
            return a[1:] == b[1:]
        if ta in ("l", "t", "S", "F"):
            return len(a[1]) == len(b[1]) and all(same(x, y) for x, y in zip(a[1], b[1]))
        if ta == "d":
            return len(a[1]) == len(b[1]) and all(same(k1, k2) and same(v1, v2)
                                                  for (k1, v1), (k2, v2) in zip(a[1], b[1]))
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return False
    return False


if __name__ == "__main__":  # self-check
    vals = [None, True, 3, 2.5, float("nan"), float("inf"), 1 + 2j, "x", b"\x00", [1, (2, 3)], {"a": {1, 2}},
            frozenset({"q"}), range(0, 5, 2), {(1, 2): [None]}]
    for v in vals:
        e = json.loads(dumps(v))
        assert same(e, encode(v)), v
        if not (isinstance(v, float) and math.isnan(v)):
            assert decode(e) == v, v
    assert same(encode(3), encode(3.0000001)) and not same(encode(3), encode(3.1))
    assert not same(encode(True), encode(1)) and not same(encode((1,)), encode([1]))
    assert not same(["?", "x"], ["?", "x"]) and not same(["T"], ["T"]) and same(["E", "ValueError"], ["E", "ValueError"])
    assert not same("garbage", encode(1)) and not same([], encode(1)) and not same(["i"], encode(1))
    print("codec ok")
