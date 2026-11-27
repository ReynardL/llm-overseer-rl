"""Greedy val eval inside the training process, on the trainer's own vLLM engine with the live LoRA.

Trajectories come from eval/evalharness.py's run_pool, Rollout and describe: the same parser, template, grader, caps
and record format as the harness, by construction. Only the engine differs. It is the trainer's vLLM engine
(Unsloth's shared-weight model.vllm_engine), woken the way TRL's colocate generate wakes it, with the LoRA request that
Unsloth's patched generate passes.

Output in `out`: traj.jsonl, meta.json (harness keys plus engine and step), runs.jsonl, and done.json once complete.
A PAUSE aborts the eval (exit 3 in the result). An unfinished earlier attempt in `out` is moved to out.aborted<N>
first, so a rerun is always a whole eval.
"""
import hashlib
import json
import os
import queue
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalharness as eh  # noqa: E402  (puts env/ on the path)
import codeenv  # noqa: E402

LORA_NAME = "val_lora"


def fresh_dir(out):
    """Move an unfinished attempt aside (kept for the record) and create an empty `out`."""
    out = Path(out)
    if out.exists():
        assert not (out / "done.json").exists(), f"{out} is a finished eval"
        n = 1
        while out.with_name(f"{out.name}.aborted{n}").exists():
            n += 1
        out.rename(out.with_name(f"{out.name}.aborted{n}"))
    out.mkdir(parents=True)
    return out


def wake(vg):
    """What TRL's VLLMGeneration.generate does before llm.generate in colocate sleep mode (vllm_generation.py L589, L704)."""
    if vg.enable_sleep_mode:
        if vg._llm_weights_sleeping:
            vg.sync_weights()
        vg.llm.wake_up(tags=["kv_cache"])


def sleep(vg):
    """What it does after (L750-752), so training could go on from the same state."""
    if vg.enable_sleep_mode:
        vg.llm.sleep(level=2)
        vg._llm_weights_sleeping = True


def run(trainer, model_name, task_ids, out, pauses, workers=32, sampling_params=None, allow_test=False):
    out = fresh_dir(out)
    test = [t for t in task_ids if codeenv._tasks[t]["split"] == "test"]
    # allow_test: run/final_test.py only, and then the list is test tasks only
    assert len(test) == len(task_ids) if allow_test else not test, f"{len(test)} of {len(task_ids)} tasks are test (allow_test={allow_test})"
    tok, tmpl = eh.make_tokenizer(model_name)  # asserts the pinned template sha
    live_tmpl = trainer.chat_template or trainer.processing_class.chat_template
    assert hashlib.sha256(live_tmpl.encode()).hexdigest() == eh.PINNED, "trainer template differs from the harness's"
    eop = eh.eos_and_pad(model_name, tok)
    vg = trainer.vllm_generation
    llm = vg.llm
    max_model_len = llm.llm_engine.model_config.max_model_len
    assert max_model_len == eh.CTX, max_model_len
    if sampling_params is None:
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, max_tokens=eh.MAX_COMPLETION, detokenize=False, **eh.DECODE["greedy"])
    # Unsloth's generate patch (unsloth/models/rl.py L4420-4437): live LoRA tensors when weights are shared, else None
    # (TRL then syncs merged weights into vLLM, which wake() does through sync_weights).
    lora = trainer.model.load_lora(LORA_NAME, load_tensors=True) if getattr(llm, "shared_weights", False) else None
    meta = {"model": model_name, "adapter": None, "decode": "greedy", "sampling": repr(sampling_params), "samples": 1,
            "tasks_sha256": hashlib.sha256("".join(t + "\n" for t in task_ids).encode()).hexdigest(),
            "template_sha256": eh.PINNED, "eos_and_pad": eop,
            "response_template_sha256": hashlib.sha256(json.dumps(tok.response_template, sort_keys=True, default=str).encode()).hexdigest(),
            "engine": "inproc", "lora": LORA_NAME if lora is not None else None, "shared_weights": getattr(llm, "shared_weights", False),
            "global_step": trainer.state.global_step, "env": {"MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    work = queue.Queue()
    for t in task_ids:
        work.put((t, 0))
    log_path, codeenv.LOG_PATH = codeenv.LOG_PATH, None  # val trajectories stay out of the training log
    t0 = time.perf_counter()
    try:
        wake(vg)
        eng = eh.Engine(llm, sampling_params, lora)
        res = eh.run_pool(eng, tok, tmpl, eop, max_model_len, work, out, pauses, workers)
    finally:
        codeenv.LOG_PATH = log_path
        sleep(vg)
    res["total_secs"] = round(time.perf_counter() - t0, 1)
    if eng.error:
        res["engine_error"] = eng.error[-2000:]
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "workers": workers, "end": time.strftime("%F %T")}) + "\n")
    if res["exit"] == 0:
        rows = [json.loads(ln) for ln in open(out / "traj.jsonl")]
        assert sorted(r["task_id"] for r in rows) == sorted(task_ids), "eval finished without every task"
        tmp = out / "done.json.tmp"
        tmp.write_text(json.dumps(res))
        os.replace(tmp, out / "done.json")
    return res
