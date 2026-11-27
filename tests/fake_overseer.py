"""CPU stand-in for overseer_call.sh, for test_orch.py: same arguments, writes claude -p's JSON shape.

  OVERSEER_CMD="python3 fake_overseer.py" with FAKE_OVERSEER=<json file>: {"<call>": action} or {"<call>.<attempt>": action}
An action is a dict, the decision to give: {"decision": "REVERT", "target": "e1", "config": {knob overrides on the root config}},
or one of these strings.
No answer (the orchestrator retries, then halts): hang (sleep past the timeout), error (exit 1, no output),
iserror (a result with is_error, as on a usage limit), claudeexit (claude -p exits 1 after a result).
An invalid answer (one retry, then the CONTINUE fallback): invalid (a decision outside the enum), badtarget (a REVERT to
an eval id that does not exist), badmodel, nomodel (empty modelUsage), turns (2 API requests), mcp (an MCP server in
init), tokens (70K input tokens). Other: die (SIGKILL the orchestrator, as a crash mid-call), clamp (a NEW_PHASE with
the learning rate out of range). Default: CONTINUE. {"long": true} at the top level makes every text field 100 words.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run"))
import knobs  # noqa: E402

prompt_f, schema_f, out = sys.argv[1:4]
name = Path(out).name  # NNN_role.raw.json
call, role = name.split(".")[0].split("_", 1)
req = json.loads(Path(out.replace(".raw.json", ".request.json")).read_text())
sc = json.loads(Path(os.environ["FAKE_OVERSEER"]).read_text()) if os.environ.get("FAKE_OVERSEER") else {}
act = sc.get(f"{int(call)}.{req['attempt']}", sc.get(str(int(call))))
log = Path(os.environ["FAKE_OVERSEER"] + ".calls") if os.environ.get("FAKE_OVERSEER") else None
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"call": int(call), "role": role, "attempt": req["attempt"], "action": act}) + "\n")
if act == "hang":
    time.sleep(60)
if act == "die":
    os.kill(os.getppid(), signal.SIGKILL)
    sys.exit(1)
if act == "error":
    sys.exit(1)
text = " ".join(["word"] * 100) if sc.get("long") else "fake"
so = {"decision": "CONTINUE", **{k: text for k in knobs.TEXT_FIELDS}}
cfg = {k: knobs.ROOT[k] for k in knobs.knob_names()}
if isinstance(act, dict):
    so.update({k: act[k] for k in ("decision", "target") if k in act})
    if act["decision"] != "CONTINUE":
        so["config"] = {**cfg, **act.get("config", {})}
elif act == "invalid":
    so["decision"] = "MAYBE"
elif act == "badtarget":
    so.update(decision="REVERT", target="e99", config=cfg)
elif act == "clamp":
    so.update(decision="NEW_PHASE", config={**cfg, "learning_rate": 1e-3})
model = "claude-opus-5-5" if act == "badmodel" else "claude-fable-5-2"
iters = [{"type": "message"}] * (2 if act == "turns" else 1)  # "turns": the overseer took a second API request
res = {"type": "result", "is_error": act == "iserror", "result": "usage limit reached" if act == "iserror" else "",
       "structured_output": None if act == "iserror" else so, "total_cost_usd": 0.01, "num_turns": 3 if act == "turns" else 2,
       "usage": {"input_tokens": 70000 if act == "tokens" else len(Path(prompt_f).read_text()) // 4, "output_tokens": 50, "iterations": iters},
       "modelUsage": {} if act == "nomodel" else {model: {}}}
init = {"type": "system", "subtype": "init", "tools": ["StructuredOutput"], "mcp_servers": [{"name": "x"}] if act == "mcp" else [],
        "claude_code_version": "fake"}
Path(out).write_text(json.dumps(res))
Path(out[:-len(".json")] + ".stream.jsonl").write_text("".join(json.dumps(x) + chr(13) + chr(10) for x in (init, res)))  # CRLF, as from claude.exe
print(f"exit={1 if act == 'claudeexit' else 0} secs=0 cwd=fake")
