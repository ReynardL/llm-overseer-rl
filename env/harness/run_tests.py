"""Runs inside the sandbox: the visible-test runner behind the run_tests tool. Output is for the agent only.

  python3 -I -S /h/run_tests.py [workdir]   (default /work; runs test_*.py functions named test*)
"""
import ast
import os
import signal
import sys
import traceback

work = sys.argv[1] if len(sys.argv) > 1 else "/work"
sys.path.insert(0, work)
sys.dont_write_bytecode = True
PER_TEST = 0.5  # references run in ms; a looping fix must not stall the (sequential) tool loop


class TestTimeout(BaseException):
    pass


def _alarm(*_):
    raise TestTimeout()


def short(v, n=200):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def where(tb):
    """Innermost frame in a workspace file other than the test file, as 'solution.py:4: <code>'."""
    best = None
    for fr in traceback.extract_tb(tb):
        name = os.path.basename(fr.filename)
        if fr.filename.startswith(work) and not name.startswith("test_"):
            best = f"{name}:{fr.lineno}: {(fr.line or '').strip()}"
    return best


def explain_assert(tb, ns):
    fr = traceback.extract_tb(tb)[-1]
    line = (fr.line or "").strip()
    msg = line if len(line) <= 300 else line[:300] + "..."  # a long expected literal is in the test file already
    try:
        node = ast.parse(line).body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            left = ast.unparse(node.test.left)
            signal.setitimer(signal.ITIMER_REAL, PER_TEST)
            try:
                got = eval(compile(left, "test", "eval"), ns)  # recompute the left side to show what it returned
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            msg += f"\n    left side returned: {short(got)}"
    except BaseException:  # noqa: BLE001
        pass
    return msg


def main():
    signal.signal(signal.SIGALRM, _alarm)
    files = sorted(f for f in os.listdir(work) if f.startswith("test_") and f.endswith(".py"))
    passed = failed = 0
    for fname in files:
        ns = {"__name__": fname[:-3], "__file__": os.path.join(work, fname)}
        try:
            signal.setitimer(signal.ITIMER_REAL, PER_TEST)
            exec(compile(open(os.path.join(work, fname)).read(), os.path.join(work, fname), "exec"), ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException as e:  # noqa: BLE001
            signal.setitimer(signal.ITIMER_REAL, 0)
            loc = where(e.__traceback__)
            print(f"ERROR collecting {fname}: {type(e).__name__}: {short(str(e), 300)}" + (f"\n    at {loc}" if loc else ""))
            failed += 1
            continue
        for name, fn in list(ns.items()):
            # only tests defined in this file: an imported `test_duplicate` from solution.py is not a test
            if not (name.startswith("test") and callable(fn) and getattr(fn, "__module__", None) == ns["__name__"]):
                continue
            try:
                signal.setitimer(signal.ITIMER_REAL, PER_TEST)
                fn()
                signal.setitimer(signal.ITIMER_REAL, 0)
                passed += 1
                print(f"{name} PASSED", flush=True)
            except TestTimeout:
                failed += 1
                print(f"{name} TIMEOUT (over {PER_TEST:g} s)", flush=True)
            except AssertionError as e:
                signal.setitimer(signal.ITIMER_REAL, 0)
                failed += 1
                print(f"{name} FAILED: {explain_assert(e.__traceback__, ns)}", flush=True)
            except BaseException as e:  # noqa: BLE001
                signal.setitimer(signal.ITIMER_REAL, 0)
                failed += 1
                loc = where(e.__traceback__)
                print(f"{name} ERROR: {type(e).__name__}: {short(str(e), 300)}" + (f"\n    at {loc}" if loc else ""),
                      flush=True)
    print(f"{failed} failed, {passed} passed")


main()
