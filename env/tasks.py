"""Task generator: seeded-bug repair tasks from MBPP (CC-BY-4.0) and MBPP+ (EvalPlus, Apache-2.0).

  python tasks.py --out ~/llm-overseer-rl/tasks/v1 [--seed 0] [--limit N] [--workers 4]

Per base function: normalize the reference with ast.unparse, collect hidden inputs (MBPP+ inputs, else type-aware
mutations of the MBPP assert arguments), record reference outputs in the sandbox (two hash seeds, drop anything that
differs, raises, times out or is slow), then make mutants and keep those that fail at least one visible and one hidden
test. Splits are disjoint by base function. Everything is seeded; outputs:
  tasks.jsonl      one task per line (workspace files, oracle, provenance, knobs). No hidden expected outputs.
  refs.jsonl       per base function: hidden inputs and expected outputs. Never enters the sandbox.
  functions.jsonl  per base function: split, counts, drop reason.
  manifest.json    seed, dataset revisions, counts.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "sandbox"), str(HERE / "harness")]
import codec  # noqa: E402
from client import Sandbox  # noqa: E402

MBPP_REV = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
PLUS_REV = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
PLUS_FILE = "data/test-00000-of-00001-d5781c9c51e02795.parquet"
MAX_HIDDEN, MIN_HIDDEN, MAX_INPUT_BYTES, MAX_REF_MS = 30, 5, 2000, 200
NONDET_IMPORTS = {"random", "time", "datetime", "uuid", "secrets", "os", "sys", "threading", "multiprocessing",
                  "subprocess", "socket", "numpy", "pandas"}
PY = ["/usr/bin/python3", "-I", "-S"]


# ---------------------------------------------------------------- data
def load_problems():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    parts = [pd.read_parquet(hf_hub_download("google-research-datasets/mbpp", f"full/{s}-00000-of-00001.parquet",
                                             repo_type="dataset", revision=MBPP_REV))
             for s in ("prompt", "test", "train", "validation")]
    mbpp = pd.concat(parts).set_index("task_id")
    plus = pd.read_parquet(hf_hub_download("evalplus/mbppplus", PLUS_FILE, repo_type="dataset", revision=PLUS_REV))
    plus = plus.set_index("task_id")
    probs = []
    for tid, r in mbpp.sort_index().iterrows():
        p = {"base": f"mbpp/{tid}", "text": r.text, "code": r.code, "tests": list(r.test_list), "imports": [],
             "setup": r.test_setup_code, "plus_inputs": None, "source": "mbpp"}
        if tid in plus.index:
            q = plus.loc[tid]
            p.update(code=q.code,  # keep MBPP's text: the MBPP+ prompt embeds an assert tests=list(q.test_list), imports=list(q.test_imports),
                     plus_inputs=parse_plus_inputs(q.test), source="mbpp+")
        probs.append(p)
    return probs


def _lit(node):
    """literal_eval that also accepts inf / nan names (MBPP+ inputs use `from math import inf`)."""
    class Sub(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in ("inf", "nan"):
                return ast.copy_location(ast.Constant(float(n.id)), n)
            return n
    return ast.literal_eval(Sub().visit(copy.deepcopy(node)))


def parse_plus_inputs(test):
    for line in test.splitlines():
        if line.startswith("inputs = "):
            try:
                return _lit(ast.parse(line).body[0].value)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                return None
    return None


# ---------------------------------------------------------------- inputs
def mutate_value(v, rng, depth=0):
    t = type(v)
    if t is bool:
        return not v
    if t is int:
        return rng.choice([v + 1, v - 1, v + 2, v * 2, -v, 0, abs(v) // 2, rng.randint(-5, 2 * abs(v) + 10)])
    if t is float:
        return rng.choice([v + 0.5, v * 1.5, -v, 0.0, round(rng.uniform(-10, 10), 2)])
    if t is str:
        if not v:
            return rng.choice(["a", "ab", " "])
        i = rng.randrange(len(v))
        return rng.choice([v[::-1], v[:i] + v[i + 1:], v + v[i], v[:i] + v[i].swapcase() + v[i + 1:],
                           "".join(rng.sample(v, len(v))), v[:i] + rng.choice(v) + v[i:]])
    if t in (list, tuple, set):
        xs = list(v)
        if xs and depth < 3:
            i = rng.randrange(len(xs))
            op = rng.randrange(6)
            if op == 0:
                xs = xs[::-1]
            elif op == 1:
                del xs[i]
            elif op == 2:
                xs.insert(i, xs[i])
            elif op == 3:
                xs[i] = mutate_value(xs[i], rng, depth + 1)
            elif op == 4:
                rng.shuffle(xs)
            else:
                xs.append(mutate_value(xs[i], rng, depth + 1))
        try:
            return t(xs)
        except TypeError:
            return v
    if t is dict and v and depth < 3:
        d = dict(v)
        k = rng.choice(list(d))
        if rng.random() < 0.5:
            d[k] = mutate_value(d[k], rng, depth + 1)
        else:
            del d[k]
        return d
    return v


def gen_inputs(seeds, rng, n=48):
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        args = list(copy.deepcopy(rng.choice(seeds)))
        for i in rng.sample(range(len(args)), rng.randint(1, min(2, len(args)))) if args else []:
            args[i] = mutate_value(args[i], rng)
        out.append(args)
    return out


def key(args):
    return json.dumps(codec.encode(list(args)))


# ---------------------------------------------------------------- mutation operators
ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.Div: ast.Mult, ast.FloorDiv: ast.Div,
         ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.BitXor: ast.BitOr,
         ast.LShift: ast.RShift, ast.RShift: ast.LShift}
CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}


def _plus(e, k):
    """e + k, folded into an existing `x + c` / `x - c` so the bug doesn't read as `n + 1 + 1`."""
    if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)) and isinstance(e.right, ast.Constant)             and type(e.right.value) is int:
        c = e.right.value * (1 if isinstance(e.op, ast.Add) else -1) + k
        return e.left if c == 0 else ast.BinOp(e.left, ast.Add() if c > 0 else ast.Sub(), ast.Constant(abs(c)))
    return ast.BinOp(e, ast.Add() if k > 0 else ast.Sub(), ast.Constant(abs(k)))


def sites(tree):
    """All (path, op, label, apply) mutation sites inside function bodies. apply(node) mutates a copied node in place
    or returns a replacement."""
    out = []

    def walk(node, path, in_fn, parent):
        in_fn = in_fn or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        if in_fn:
            out.extend((path, *s) for s in node_sites(node, parent))
        for field, val in ast.iter_fields(node):
            if isinstance(val, list):
                for i, x in enumerate(val):
                    if isinstance(x, ast.AST):
                        walk(x, path + [(field, i)], in_fn, node)
            elif isinstance(val, ast.AST):
                walk(val, path + [(field, None)], in_fn, node)

    walk(tree, [], False, None)
    return out


def node_sites(n, parent):
    s = []
    if isinstance(n, ast.BinOp) and type(n.op) in ARITH:
        new = ARITH[type(n.op)]
        s.append(("arith", f"{type(n.op).__name__}->{new.__name__}", lambda m, new=new: setattr(m, "op", new()) or m))
    if isinstance(n, ast.AugAssign) and type(n.op) in ARITH:
        new = ARITH[type(n.op)]
        s.append(("aug", f"{type(n.op).__name__}=->{new.__name__}=", lambda m, new=new: setattr(m, "op", new()) or m))
    if isinstance(n, ast.Compare):
        for i, op in enumerate(n.ops):
            if type(op) in CMP:
                new = CMP[type(op)]

                def f(m, i=i, new=new):
                    m.ops[i] = new()
                    return m
                s.append(("cmp", f"{type(op).__name__}->{new.__name__}", f))
    if isinstance(n, ast.BoolOp):
        new = ast.Or if isinstance(n.op, ast.And) else ast.And
        s.append(("bool", f"{type(n.op).__name__}->{new.__name__}", lambda m, new=new: setattr(m, "op", new()) or m))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
        s.append(("not", "drop not", lambda m: m.operand))
    if isinstance(n, ast.Constant) and type(n.value) is int:
        for k in (1, -1):
            if n.value + k == 0 and isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub)):
                continue  # `i - 0` reads as a planted bug
            s.append(("const", f"{n.value}->{n.value + k}", lambda m, k=k: ast.Constant(m.value + k)))
    if isinstance(n, ast.Constant) and type(n.value) is bool:
        s.append(("const", f"{n.value}->{not n.value}", lambda m: ast.Constant(not m.value)))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        if n.func.id in ("min", "max"):
            new = "max" if n.func.id == "min" else "min"
            s.append(("minmax", f"{n.func.id}->{new}", lambda m, new=new: setattr(m.func, "id", new) or m))
        if n.func.id == "range" and n.args and not isinstance(n.args[-1 if len(n.args) < 3 else 1], ast.Constant):
            j = len(n.args) - 1 if len(n.args) < 3 else 1
            for k in (1, -1):
                def f(m, j=j, k=k):
                    m.args[j] = _plus(m.args[j], k)
                    return m
                s.append(("range", f"range stop {k:+d}", f))
    if isinstance(n, ast.Call) and len(n.args) >= 2 and not any(isinstance(a, ast.Starred) for a in n.args[:2]):
        if ast.dump(n.args[0]) != ast.dump(n.args[1]):
            def f(m):
                m.args[0], m.args[1] = m.args[1], m.args[0]
                return m
            s.append(("args", "swap args 1,2", f))
    if isinstance(n, ast.Slice):
        for field in ("lower", "upper"):
            e = getattr(n, field)
            if e is not None and not isinstance(e, ast.Constant):
                for k in (1, -1):
                    def f(m, field=field, k=k):
                        setattr(m, field, _plus(getattr(m, field), k))
                        return m
                    s.append(("slice", f"slice {field} {k:+d}", f))
    return s


def apply(tree, muts):
    t = copy.deepcopy(tree)
    for path, _, _, fn in sorted(muts, key=lambda m: -len(m[0])):  # deepest first, so outer paths stay valid
        parent, node = None, t
        for field, i in path:
            parent, node = node, getattr(node, field)[i] if i is not None else getattr(node, field)
        new = fn(node)
        if new is not node:
            field, i = path[-1]
            if i is None:
                setattr(parent, field, new)
            else:
                getattr(parent, field)[i] = new
    return ast.fix_missing_locations(t)


# ---------------------------------------------------------------- sandbox helpers
class Runner:
    def __init__(self, sb, tmp):
        self.sb, self.tmp = sb, tmp

    def _dir(self, files):
        d = tempfile.mkdtemp(dir=self.tmp)
        for name, text in files.items():
            Path(d, name).write_text(text)
        return d

    def outputs(self, src, entry, inputs, hashseed="0", timeout=30, stop=False):
        d = self._dir({"solution.py": src, "inputs.json": json.dumps([codec.encode(list(a)) for a in inputs])})
        try:
            r = self.sb.run(PY + ["/h/grade.py", "/work/solution.py", entry, "/work/inputs.json", "1.0"] + (["stop"] if stop else []), work=d,
                            timeout=timeout, cap=4 << 20, env={"PYTHONHASHSEED": hashseed})
        finally:
            rmtree(d)
        res = {}
        for ln in r["stdout"].splitlines():
            if ln.startswith("R "):
                _, i, js = ln.split(" ", 2)
                res.setdefault(int(i), json.loads(js))
        return res, r

    def visible(self, src, test_src):
        d = self._dir({"solution.py": src, "test_solution.py": test_src})
        try:
            r = self.sb.run(PY + ["/h/run_tests.py"], work=d, timeout=15)
        finally:
            rmtree(d)
        return {m.group(1): m.group(2) for m in re.finditer(r"^(test\w*) (PASSED|FAILED|ERROR|TIMEOUT)", r["stdout"], re.M)}, r


def rmtree(d):
    for f in os.listdir(d):
        os.unlink(os.path.join(d, f))
    os.rmdir(d)


def graded_pass(res, expected):
    """All hidden inputs pass: result and (if the reference mutated them) the arguments match."""
    for i, exp in enumerate(expected):
        got = res.get(i)
        if got is None or not codec.same(got.get("r"), exp["r"]):
            return False
        if ("a" in exp) != ("a" in got) or ("a" in exp and not codec.same(got["a"], exp["a"])):
            return False
    return True


# ---------------------------------------------------------------- per function
def test_file(entry, extra_names, imports, asserts):
    head = list(imports) + (["import math"] if any("math." in a for a in asserts) and "import math" not in imports else [])
    names = ", ".join([entry] + sorted(extra_names))
    body = "".join(f"\n\ndef test_{i + 1}():\n    {a}\n" for i, a in enumerate(asserts))
    return "\n".join(head + [f"from solution import {names}"]) + "\n" + body


def prepare(p, run, rng):
    """Validate one base function. Returns (info, ref) where ref is None if dropped."""
    info = {"base": p["base"], "source": p["source"]}
    if p["setup"]:
        return {**info, "drop": "test_setup_code"}, None
    try:
        tree = ast.parse(p["code"].replace("\r\n", "\n"))
    except SyntaxError:
        return {**info, "drop": "syntax"}, None
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or "")])}
    if mods & NONDET_IMPORTS:
        return {**info, "drop": "nondet_import"}, None
    defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    top = defs | {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    calls = Counter()
    seeds, extra = [], set()
    for a in p["tests"]:
        try:
            at = ast.parse(a).body[0]
        except SyntaxError:
            return {**info, "drop": "assert_syntax"}, None
        extra |= {n.id for n in ast.walk(at) if isinstance(n, ast.Name) and n.id in top}
        cs = [n for n in ast.walk(at) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs]
        calls.update(c.func.id for c in cs)
        seeds.append(cs)
    if not calls:
        return {**info, "drop": "no_entry_call"}, None
    entry = calls.most_common(1)[0][0]
    extra.discard(entry)
    seed_args = []
    for cs in seeds:
        c = next((c for c in cs if c.func.id == entry), None)
        if c is None or c.keywords:
            return {**info, "drop": "seed_unparsable"}, None
        try:
            seed_args.append([_lit(x) for x in c.args])
        except (ValueError, TypeError, SyntaxError):
            return {**info, "drop": "seed_unparsable"}, None
    src = ast.unparse(tree) + "\n"
    tests = test_file(entry, extra, p["imports"], p["tests"])
    vis, r = run.visible(src, tests)
    if len(vis) != len(p["tests"]) or set(vis.values()) != {"PASSED"}:
        return {**info, "drop": "reference_fails_visible", "detail": r["stdout"][-300:]}, None
    seen = {key(a) for a in seed_args}
    cands = []
    pool = list(p["plus_inputs"] or []) + gen_inputs(seed_args, rng)
    for a in pool:
        if not isinstance(a, (list, tuple)):
            continue
        k = key(a)
        if k in seen or len(k) > MAX_INPUT_BYTES or "\"?\"" in k:
            continue
        seen.add(k)
        cands.append(list(a))
    if p["plus_inputs"]:  # prefer MBPP+ inputs, then generated ones, each group shuffled
        n_plus = sum(1 for a in cands if key(a) in {key(x) for x in p["plus_inputs"] if isinstance(x, (list, tuple))})
        head, tail = cands[:n_plus], cands[n_plus:]
        rng.shuffle(head)
        rng.shuffle(tail)
        cands = head + tail
    else:
        rng.shuffle(cands)
    cands = cands[:120]
    r0, raw0 = run.outputs(src, entry, cands, "0")
    r1, _ = run.outputs(src, entry, cands, "1")
    keep, expected, why = [], [], Counter()
    for i, a in enumerate(cands):
        x, y = r0.get(i), r1.get(i)
        if x is None or y is None:
            why["missing"] += 1
        elif x["r"][0] in ("E", "T", "?"):
            why[x["r"][0]] += 1
        elif json.dumps(x["r"]) != json.dumps(y["r"]) or json.dumps(x.get("a")) != json.dumps(y.get("a")):
            why["nondet"] += 1
        elif x["ms"] > MAX_REF_MS:
            why["slow"] += 1
        else:
            keep.append(a)
            expected.append({k: v for k, v in x.items() if k != "ms"})
        if len(keep) >= MAX_HIDDEN:
            break
    info.update(entry=entry, n_candidates=len(cands), hidden=len(keep), rejected=dict(why))
    if why["nondet"]:
        return {**info, "drop": "nondeterministic"}, None
    if len(keep) < MIN_HIDDEN:
        return {**info, "drop": "too_few_hidden"}, None
    ref = {"base": p["base"], "entry": entry, "src": src, "tree": tree, "tests": p["tests"], "test_src": tests,
           "extra": sorted(extra), "imports": p["imports"], "seed_args": seed_args, "text": p["text"],
           "hidden": {"inputs": [codec.encode(list(a)) for a in keep], "expected": expected}}
    return info, ref


def mutants(ref, run, rng, n_single=16, n_double=8):
    ss = sites(ref["tree"])
    rng.shuffle(ss)
    groups = {}
    for x in ss:
        groups.setdefault(x[1], []).append(x)
    ss = []  # round-robin over operator classes, so `const` (two sites per literal) doesn't dominate
    while any(groups.values()):
        for g in groups.values():
            if g:
                ss.append(g.pop())
    combos = [[s] for s in ss[:n_single]]
    pairs = 0
    for _ in range(n_double * 4):
        if pairs >= n_double or len(ss) < 2:
            break
        a, b = rng.sample(ss, 2)
        if a[0][:len(b[0])] == b[0] or b[0][:len(a[0])] == a[0]:  # nested sites: skip
            continue
        combos.append([a, b])
        pairs += 1
    out, seen, looping, quiet = [], {ref["src"]}, 0, []
    exp = ref["hidden"]["expected"]
    inputs = [codec.decode(e) for e in ref["hidden"]["inputs"]]
    for muts in combos:
        try:
            src = ast.unparse(apply(ref["tree"], muts)) + "\n"
            compile(src, "solution.py", "exec")
        except Exception:  # noqa: BLE001 (an operator produced invalid code)
            continue
        if src in seen:
            continue
        seen.add(src)
        vis, _ = run.visible(src, ref["test_src"])
        fails = [k for k, v in vis.items() if v != "PASSED"]
        if not fails and len(vis) == len(ref["tests"]):  # passes every visible assert: a candidate for recover()
            quiet.append({"src": src, "muts": [{"op": m[1], "change": m[2]} for m in muts]})
        if not fails or len(vis) != len(ref["tests"]):
            continue
        if "TIMEOUT" in vis.values():  # looping mutants: every run_tests call would cost the full timeout
            looping += 1
            continue
        res, _ = run.outputs(src, ref["entry"], inputs, timeout=20, stop=True)
        if graded_pass(res, exp):
            continue  # equivalent on hidden tests
        out.append({"src": src, "fails": sorted(fails), "muts": [{"op": m[1], "change": m[2]} for m in muts]})
    return out, len(combos), looping, quiet


def literal_assert(entry, args, exp):
    """`assert entry(args) == expected` if every value survives repr + literal_eval and the line is short, else None."""
    vals = list(args) + [codec.decode(exp["r"])]
    try:
        if not all(codec.same(codec.encode(ast.literal_eval(repr(v))), codec.encode(v)) for v in vals):
            return None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):  # nan, inf, set(), objects
        return None
    line = f"assert {entry}({', '.join(map(repr, args))}) == {vals[-1]!r}"
    return line if len(line) <= 200 else None


def recover(ref, quiet, run):
    """v2, for a function with no valid mutant: promote one hidden input to a 4th visible assert that some quiet
    mutants (they pass all 3 MBPP asserts) fail, and drop it from the hidden set. A kept mutant must fail the new
    assert without a timeout and still fail another hidden input. Updates ref in place; returns mutants in the
    mutants() format, or [] (ref untouched)."""
    exp = ref["hidden"]["expected"]
    inputs = [codec.decode(e) for e in ref["hidden"]["inputs"]]
    if len(inputs) <= MIN_HIDDEN or not quiet:
        return []
    failing = []
    for q in quiet:
        res, _ = run.outputs(q["src"], ref["entry"], inputs, timeout=20, stop=True)
        f = {i for i, e in enumerate(exp) if not graded_pass({0: res[i]} if i in res else {}, [e])}
        if len(f) >= 2:
            failing.append((q, f))
    cands = []
    for i, a in enumerate(inputs):
        line = literal_assert(ref["entry"], a, exp[i])
        n = sum(i in f for _, f in failing)
        if line and n:
            cands.append((-n, len(line), i, line))
    if not cands:
        return []
    _, _, h, line = min(cands)  # the input most mutants fail, then the shortest assert
    tests = ref["tests"] + [line]
    test_src = test_file(ref["entry"], ref["extra"], ref["imports"], tests)
    vis, _ = run.visible(ref["src"], test_src)
    if len(vis) != len(tests) or set(vis.values()) != {"PASSED"}:
        return []
    out = []
    for q, f in failing:
        if h not in f:
            continue
        vis, _ = run.visible(q["src"], test_src)
        fails = [k for k, v in vis.items() if v != "PASSED"]
        if len(vis) == len(tests) and fails and "TIMEOUT" not in vis.values():
            out.append({"src": q["src"], "fails": sorted(fails), "muts": q["muts"]})
    if out:
        keep = [i for i in range(len(inputs)) if i != h]
        ref["hidden"] = {"inputs": [ref["hidden"]["inputs"][i] for i in keep], "expected": [exp[i] for i in keep]}
        ref.update(tests=tests, test_src=test_src, seed_args=ref["seed_args"] + [list(inputs[h])],
                   promoted={"hidden_index": h, "assert": line})
    return out


def apply_edits(text, edits):
    for old, new in edits:
        if text.count(old) != 1:
            return None
        text = text.replace(old, new)
    return text


def oracle_edits(bug, fix):
    """Line-level (old, new) edits that turn bug into fix, each `old` unique when applied in order. Falls back to one
    edit spanning every changed line."""
    a, b = bug.splitlines(keepends=True), fix.splitlines(keepends=True)
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    edits = []
    for _, i1, i2, j1, j2 in ops:
        lo, hi = i1, max(i2, i1 + 1)
        while lo > 0 or hi < len(a):
            if a[lo:hi] and bug.count("".join(a[lo:hi])) == 1:
                break
            lo, hi = max(0, lo - 1), min(len(a), hi + 1)
        edits.append(("".join(a[lo:hi]), "".join(a[lo:i1]) + "".join(b[j1:j2]) + "".join(a[i2:hi])))
    if apply_edits(bug, edits) == fix:
        return edits
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    edits = [("".join(a[i1:max(i2, i1 + 1)]), "".join(b[j1:j2]) + "".join(a[i2:max(i2, i1 + 1)]))]
    assert apply_edits(bug, edits) == fix, "oracle edit fallback failed"
    return edits


def build_tasks(ref, muts, distractor_pool, rng, per_fn):
    rng.shuffle(muts)
    singles = [m for m in muts if len(m["muts"]) == 1]
    doubles = [m for m in muts if len(m["muts"]) == 2]
    chosen = (singles[: max(per_fn - 2, per_fn - len(doubles))] + doubles)[:per_fn]
    tasks = []
    for j, m in enumerate(chosen):
        n_dis = rng.choice([0, 0, 1, 2])
        dis = [d for d in rng.sample(distractor_pool, min(len(distractor_pool), 6))
               if not (d["names"] & ({n.name for n in ref["tree"].body if hasattr(n, "name")} | {ref["entry"]}))][:n_dis]
        k = rng.choice([1, 2, 3])
        failing = [int(t.split("_")[1]) - 1 for t in m["fails"]]
        first = rng.choice(failing)
        others = [i for i in range(len(ref["tests"])) if i != first]
        vis = sorted([first] + rng.sample(others, min(k - 1, len(others))))
        parts = [d["src"] for d in dis]
        pos = rng.randrange(len(parts) + 1)
        imports = sorted({ln for d in dis for ln in d["imports"]})

        def module(core, parts=parts, pos=pos, imports=imports):
            chunks = parts[:pos] + [core] + parts[pos:]
            return ("\n".join(imports) + "\n\n" if imports else "") + "\n\n".join(c.strip("\n") for c in chunks) + "\n"
        bug, fix = module(m["src"]), module(ref["src"])
        asserts = [ref["tests"][i] for i in vis]
        tasks.append({
            "task_id": f"{ref['base'].replace('/', '-')}-{j}", "base": ref["base"], "entry": ref["entry"],
            "description": ref["text"],
            "files": {"solution.py": bug, "test_solution.py": test_file(ref["entry"], ref["extra"], ref["imports"], asserts)},
            "target_files": ["solution.py"], "oracle": {"solution.py": fix}, "oracle_edits": oracle_edits(bug, fix),
            "mutations": m["muts"], "visible_idx": vis,
            "visible_args": [codec.encode(ref["seed_args"][i]) for i in vis],
            "knobs": {"mutations": len(m["muts"]), "distractors": len(dis), "visible_tests": len(vis)},
            "distractors": [d["base"] for d in dis], "n_hidden": len(ref["hidden"]["inputs"]),
        })
    return tasks


def distractor_of(ref):
    body = ref["tree"].body
    imps = [ast.unparse(n) for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
    defs = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(defs) != 1 or len(imps) + 1 != len(body):  # only simple modules: imports + one function
        return None
    return {"base": ref["base"], "src": ast.unparse(defs[0]), "imports": imps, "names": {defs[0].name}}


def split_of(base, seed, frac_test, frac_val):
    h = int(hashlib.sha256(f"{seed}:{base}".encode()).hexdigest()[:12], 16) / 16**12
    return "test" if h < frac_test else "val" if h < frac_test + frac_val else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-fn", type=int, default=6)
    ap.add_argument("--frac-test", type=float, default=0.25)
    ap.add_argument("--frac-val", type=float, default=0.125)
    ap.add_argument("--no-recover", dest="recover", action="store_false")  # v1 behavior
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    probs = load_problems()
    if a.limit:
        probs = probs[: a.limit]
    sb = Sandbox()
    tmp = tempfile.mkdtemp(dir=sb.base)
    run = Runner(sb, tmp)

    def rng_for(base, what):
        return random.Random(f"{a.seed}:{base}:{what}")

    done = Counter()

    def progress(stage, n, x):
        done[stage] += 1  # Counter += from worker threads is fine under the GIL for a progress print
        if done[stage] % 50 == 0 or done[stage] == n:
            print(f"{stage} {done[stage]}/{n}", flush=True)
        return x

    with ThreadPoolExecutor(a.workers) as ex:
        prepared = list(ex.map(lambda p: progress("prepare", len(probs), prepare(p, run, rng_for(p["base"], "inputs"))),
                               probs))
    refs = [r for _, r in prepared if r]
    infos = {i["base"]: i for i, _ in prepared}
    for r in refs:
        infos[r["base"]]["split"] = split_of(r["base"], a.seed, a.frac_test, a.frac_val)
    with ThreadPoolExecutor(a.workers) as ex:
        muts = list(ex.map(lambda r: progress("mutants", len(refs), mutants(r, run, rng_for(r["base"], "mutants"))),
                           refs))
    pools = {}
    for r in refs:
        d = distractor_of(r)
        if d:
            pools.setdefault(infos[r["base"]]["split"], []).append(d)
    tasks = []
    for r, (ms, n_tried, n_loop, quiet) in zip(refs, muts):
        info = infos[r["base"]]
        info.update(mutants_tried=n_tried, mutants_valid=len(ms), mutants_looping=n_loop)
        if not ms and a.recover:
            ms = recover(r, quiet, run)
            if ms:
                info.update(recovered=r["promoted"], hidden=len(r["hidden"]["inputs"]), mutants_valid=len(ms))
        if not ms:
            info["drop"] = "no_valid_mutant"
            continue
        pool = [d for d in pools.get(info["split"], []) if d["base"] != r["base"]]
        ts = build_tasks(r, ms, pool, rng_for(r["base"], "tasks"), a.per_fn)
        for t in ts:
            t["split"] = info["split"]
            t["recovered"] = "promoted" in r  # visible test 4 (index 3) came from a hidden input
        info["tasks"] = len(ts)
        tasks += ts
    with open(out / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    with open(out / "refs.jsonl", "w") as f:
        for r in refs:
            if infos[r["base"]].get("tasks"):
                f.write(json.dumps({"base": r["base"], "entry": r["entry"], **r["hidden"]}) + "\n")
    with open(out / "functions.jsonl", "w") as f:
        for i in infos.values():
            f.write(json.dumps(i) + "\n")
    counts = {s: {"functions": sum(1 for i in infos.values() if i.get("split") == s and i.get("tasks")),
                  "tasks": sum(1 for t in tasks if t["split"] == s)} for s in ("train", "val", "test")}
    manifest = {"seed": a.seed, "mbpp_revision": MBPP_REV, "mbppplus_revision": PLUS_REV, "per_fn": a.per_fn,
                "frac_test": a.frac_test, "frac_val": a.frac_val, "problems": len(probs),
                "drops": Counter(i.get("drop", "kept") for i in infos.values()), "splits": counts,
                "knobs": {k: Counter(str(t["knobs"][k]) for t in tasks) for k in ("mutations", "distractors", "visible_tests")},
                "operators": Counter(m["op"] for t in tasks for m in t["mutations"]),
                "recovered": {"functions": sum(1 for i in infos.values() if i.get("recovered") and i.get("tasks")),
                              "tasks": sum(1 for t in tasks if t["recovered"])},
                "mutants": {k: sum(i.get(f"mutants_{k}", 0) for i in infos.values()) for k in ("tried", "valid", "looping")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    os.rmdir(tmp)
    sb.close()


if __name__ == "__main__":
    main()
