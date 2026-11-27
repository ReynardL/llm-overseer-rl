"""Runs inside the sandbox: call the entry function on each hidden input and print the encoded results.

  python3 -I -S /h/grade.py <solution.py> <entry> <inputs.json> [per-input timeout s] [stop]

Prints one line per input: R <idx> <json {"r": result, "a": args after the call if they changed, "ms": time}>.
With `stop`, the first timeout ends the run (a timeout already fails the task; this caps a looping solution's cost).
Expected outputs never enter the sandbox; the grader compares outside. Nothing printed here is trusted.
"""
import json
import os
import signal
import sys
import time

sys.path.insert(0, "/h")
import codec  # noqa: E402

src, entry, inp = sys.argv[1:4]
per = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
stop = sys.argv[5:6] == ["stop"]
inputs = json.load(open(inp))
out = os.dup(1)  # write results to a private copy of stdout, so print() in agent code can't interleave mid-line


class _Timeout(BaseException):
    pass


def _alarm(*_):
    raise _Timeout()


def emit(idx, rec):
    os.write(out, f"R {idx} {json.dumps(rec)}\n".encode())


ns = {"__name__": "solution", "__file__": src}
try:
    code = compile(open(src).read(), "solution.py", "exec")
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, per)
    exec(code, ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
    fn = ns[entry]
except BaseException as e:
    os.write(out, f"LOADERR {type(e).__name__}\n".encode())
    sys.exit(0)

for idx, enc in enumerate(inputs):
    args = codec.decode(enc)
    before = codec.dumps(args)
    t = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, per)
        r = json.loads(codec.dumps(fn(*args)))
    except _Timeout:
        r = ["T"]
    except BaseException as e:  # noqa: BLE001 (SystemExit, KeyboardInterrupt included)
        r = ["E", type(e).__name__]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rec = {"r": r, "ms": round((time.perf_counter() - t) * 1000, 2)}
    try:
        after = codec.dumps(args)
        if after != before:
            rec["a"] = json.loads(after)
    except BaseException:  # noqa: BLE001
        rec["a"] = ["?", "args"]
    emit(idx, rec)
    if stop and r == ["T"]:
        break
