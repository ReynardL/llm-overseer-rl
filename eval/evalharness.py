"""Multi-turn eval and calibration harness: training's GRPO rollout, run on a standalone vLLM engine.

  python evalharness.py --model unsloth/Qwen3-1.7B --tasks <file, one task id per line> --out ~/llm-overseer-rl/runs/eval/<name>
         [--samples 8] [--decode train|greedy, default below] [--max-completion 3472] [--resume-cap-from N] [--adapter <LoRA dir>] [--util 0.84] [--workers 32] [--allow-test]

Identical to training by construction: each trajectory runs TRL's own GRPOTrainer code (_tokenize_prompts,
_tool_call_loop, _get_tool_suffix_ids, parse_response), called unbound on a stand-in that carries the trainer's settings.
Prompt rendering, tool-call parsing, the malformed-call fallback, tool errors, the 10-round limit
and the 3472/4096 caps are therefore TRL's. Unsloth's compiled trainer has those methods AST-identical.
Only generation is ours: each turn is one request on a shared vLLM engine, so turns from all in-flight trajectories batch
continuously. Env, tools and grader: env/codeenv.py. One worker thread per in-flight trajectory, so sandbox calls overlap.

Output: <out>/traj.jsonl, one line per finished trajectory. A restart skips finished (task_id, sample) pairs.
PAUSE: <out>/PAUSE or eval/PAUSE stops the run; in-flight trajectories are dropped and redone on resume.
Exit: 0 done, 3 paused, 2 too many errors.
--decode default: greedy when --tasks has no train task (val/test evals), else train (calibration, pilots).
--resume-cap-from N: the one documented resume exception, see check_resume.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__" and os.environ.get("MALLOC_ARENA_MAX") != "2":
    # glibc reads it only at process start, so re-exec: 32 worker threads on 2 malloc arenas hold ~3 GB less host RAM
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.execv(sys.executable, sys.orig_argv)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "env"))

import trl.chat_template_utils as tcu  # noqa: E402
import trl.trainer.grpo_trainer as gt  # noqa: E402 (loaded before the patch so Unsloth rebinds its names too)
from transformers import AutoTokenizer, GenerationConfig  # noqa: E402

# Unsloth's TRL template patch, loaded standalone (`import unsloth` touches the GPU).
_spec = importlib.util.spec_from_file_location(
    "_unsloth_trl_tool_templates", sys.prefix + "/lib/python3.12/site-packages/unsloth/models/_trl_tool_templates.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_m.patch_trl_tool_chat_templates()

import codeenv  # noqa: E402

PINNED = "1d944ff8f268b611abb296cdd24d0f51981eef1c8647ac321c3a0258f61eb6c9"  # sha256 of the chat template, as run/trainer.py pins it
TURNS, MAX_COMPLETION, CTX = 10, 3472, 4096  # was 3584; the trainer's max_tool_calling_iterations, max_completion_length, vllm_max_model_length
# vLLM SamplingParams. "train" is what TRL's colocate path sends under GRPOConfig defaults (vllm_generation.py L662-671).
DECODE = {"train": dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0),
          "greedy": dict(temperature=0.0)}
CALL_RE = re.compile(r'"name"\s*:\s*"?(view_file|edit_file|run_tests|submit)|\b(view_file|edit_file|run_tests|submit)\s*\(')


class Aborted(Exception):
    pass


def make_tokenizer(model):
    """The trainer's processing class and template, as GRPOTrainer.__init__ sets them (grpo_trainer.py L737-756)."""
    tok = AutoTokenizer.from_pretrained(model)
    if getattr(tok, "response_template", None) is None and getattr(tok, "response_schema", None) is None:
        tok = gt.add_response_schema(tok)
    tmpl = None if gt.is_chat_template_prefix_preserving(tok) else gt.get_training_chat_template(tok)
    sha = hashlib.sha256((tmpl or tok.chat_template).encode()).hexdigest()
    assert sha == PINNED, f"chat template sha256 {sha} is not the pinned {PINNED}"
    assert tcu._SUPPORTS_RESPONSE_TEMPLATE and getattr(tok, "response_template", None) is not None, "no response parser"
    return tok, tmpl


def eos_and_pad(model, tok):
    """GRPOTrainer.__init__ L398-405 (eos_token_ids) plus the pad id, as the truncation test uses them."""
    eos = GenerationConfig.from_pretrained(model).eos_token_id
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    if tok.eos_token_id not in eos:
        eos = [tok.eos_token_id, *eos]
    return [*eos, tok.pad_token_id]


def tool_methods(env):
    """The env's tools, in TRL's order (_generate_and_score_completions L2296-2300; __init__ L668 for the schema)."""
    return [getattr(env, n) for n, _ in inspect.getmembers(type(env), predicate=inspect.isfunction)
            if n not in ("reset", "get_reward") and not n.startswith("_")]


class Rollout:
    """Stand-in for GRPOTrainer: the attributes its rollout methods read, set as run/trainer.py configures them."""
    _tokenize_prompts = gt.GRPOTrainer._tokenize_prompts
    _get_tool_suffix_ids = gt.GRPOTrainer._get_tool_suffix_ids
    _tool_call_loop = gt.GRPOTrainer._tool_call_loop

    def __init__(self, tok, tmpl, tools, gen, max_model_len):
        self.processing_class = self._tokenizer = tok
        self.chat_template, self.chat_template_kwargs = tmpl, {"enable_thinking": False}
        self._is_vlm, self.tools = False, tools
        self.environment_factories, self._batch_environments, self._env_tools = {None: codeenv.CodingEnv}, [None], {None: tools}
        self.max_tool_calling_iterations, self.max_completion_length = TURNS, MAX_COMPLETION
        self.use_vllm, self.vllm_mode = True, "colocate"
        mc = SimpleNamespace(max_model_len=max_model_len)
        self.vllm_generation = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(model_config=mc)))
        self.gen, self.turns = gen, []

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        assert len(prompt_ids) == 1 and num_generations == 1 and images is None and not multimodal_fields
        ids, fin = self.gen(prompt_ids[0])
        self.turns.append({"prompt": prompt_ids[0], "ids": ids, "fin": fin})
        return [ids], None  # no logprobs: the loop only carries them along

    def run(self, env, task_id):
        """One trajectory, as GRPOTrainer._generate_and_score_completions (L2286-2325) and _generate (L2124-2185) run a row."""
        row = {"prompt": [{"role": "user", "content": ""}], "task_id": task_id}  # codeenv.dataset's row
        self._sync_tool_dicts, self._async_tool_dicts = [{f.__name__: f for f in tool_methods(env)}], [{}]
        obs = env.reset(**row)
        prompt = row["prompt"]
        prompts = [prompt[:-1] + [{**prompt[-1], "content": prompt[-1]["content"] + obs}]]
        self.turns = []
        prompt_ids, images, mm = self._tokenize_prompts(prompts)
        completion_ids, logprobs = self._generate_single_turn(prompt_ids, images, mm, 1)
        completions = [[gt.parse_response(self._tokenizer, ids, prefix=prompt_ids[i])] for i, ids in enumerate(completion_ids)]
        tool_mask, completions, completion_ids, _, n_calls, n_fail, _ = self._tool_call_loop(
            prompts, prompt_ids, completion_ids, completions, logprobs, images, mm)
        return prompt_ids[0], completion_ids[0], tool_mask[0], completions[0], n_calls, n_fail


def parse_error(tok, ids, prefix):
    """Why trl's parse_response fell back to plain text on this turn, or None if it parsed (mirrors its try block)."""
    try:
        p = tok.parse_response(ids, prefix=prefix)
        if p is None:
            return "parse_response returned None"
        if "tool_calls" in p:
            tcu._validate_tool_calls(copy.deepcopy(p["tool_calls"]))
    except (ValueError, TypeError) as e:
        return f"{type(e).__name__}: {e}"[:300]
    return None


def is_truncated(pids, cids, eop, max_model_len):
    """Unsloth's test (unsloth_compiled_cache/UnslothGRPOTrainer.py L816-850): cut by a length limit, no EOS/pad last."""
    return cids[-1] not in eop and (len(cids) >= MAX_COMPLETION or len(pids) + len(cids) >= max_model_len)


def distinct_ratio(ids, n=16):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def describe(tok, r, env, out, eop, max_model_len):
    """The trajectory's JSONL record."""
    pids, cids, mask, msgs, n_calls, n_fail = out
    asst = iter(m for m in msgs if m["role"] == "assistant")
    turns = []
    for t in r.turns:
        d = {"p": len(t["prompt"]), "g": len(t["ids"]), "fin": t["fin"]}
        m = next(asst, None) if t["ids"] else None
        if t["ids"] and m is None:
            # a tool result left the completion at exactly MAX_COMPLETION: TRL cut this turn to 0 tokens and kept no
            # message for it ("{} if post-tool completions completely truncated"); only prompts under 624 reach this
            d["cut_to_empty"] = True
        if m is not None:
            calls = m.get("tool_calls") or []
            d["calls"] = [(c.get("function") or c).get("name") for c in calls]
            err = parse_error(tok, t["ids"], t["prompt"])
            if err:
                d["parse_error"] = err
            if err and calls:
                d["mismatch"] = True  # parse_error disagrees with TRL's result: a harness bug if it ever shows
            if not calls and CALL_RE.search(m["content"]):
                d["call_like_text"] = True
            if err or d.get("call_like_text"):
                d["text"] = m["content"][:1500]
            if d["g"] >= 64:
                d["distinct16"] = round(distinct_ratio(t["ids"]), 3)
        turns.append(d)
    tools = []
    for m in msgs:
        if m["role"] == "tool":
            c = m["content"] if isinstance(m["content"], str) else str(m["content"])
            kind = "exc" if c.startswith("{'error': ") else "env_error" if c.startswith("Error:") else "ok"
            tools.append({"name": m["name"], "kind": kind, **({"out": c[:300]} if kind != "ok" else {})})
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), {})
    trunc = is_truncated(pids, cids, eop, max_model_len)
    end = ("truncated" if trunc else "turn_limit" if last.get("tool_calls") and len(r.turns) == TURNS + 1
           else "ctx_rollback" if last.get("tool_calls") else "stop")
    g = env.future.result() if env.future is not None else None
    return {"end": end, "truncated": trunc, "turns": len(r.turns), "tool_calls": n_calls, "tool_failures": n_fail,
            "prompt_tokens": len(pids), "completion_tokens": len(cids), "gen_tokens": sum(mask),
            "completion_sha": hashlib.sha256(json.dumps(list(cids)).encode()).hexdigest()[:16],  # run-to-run agreement checks
            "submitted": env.submitted, "grade": g, "turn_info": turns, "tool_results": tools,
            "calls": [{k: c[k] for k in ("tool", "args", "secs")} for c in env.rec["calls"]],
            "final_text": (last.get("content") or "")[:300]}


class Engine:
    """Owns the vLLM engine (not thread-safe). Workers block in generate(); this thread adds and steps requests."""

    def __init__(self, llm, sp, lora):
        self.eng, self.sp, self.lora = llm.llm_engine, sp, lora
        self.q, self.futs, self.ids = queue.Queue(), {}, itertools.count()
        self.stop, self.gen_tokens, self.prompt_tokens, self.requests = threading.Event(), 0, 0, 0
        self.dead, self.error = False, None

    def generate(self, ids):
        f = Future()
        self.q.put((ids, f))
        while True:
            try:
                return f.result(timeout=2)
            except TimeoutError:
                if self.dead:  # put after the loop drained the queue: nobody will answer
                    raise Aborted from None

    def loop(self):
        try:
            while not self.stop.is_set():
                busy = self.eng.has_unfinished_requests()
                try:
                    item = self.q.get_nowait() if busy else self.q.get(timeout=0.2)
                    while True:
                        ids, f = item
                        rid = str(next(self.ids))
                        self.eng.add_request(rid, {"prompt_token_ids": ids}, self.sp, lora_request=self.lora)
                        self.futs[rid] = f
                        self.prompt_tokens += len(ids)
                        self.requests += 1
                        item = self.q.get_nowait()
                except queue.Empty:
                    pass
                if self.eng.has_unfinished_requests():
                    for o in self.eng.step():
                        if o.finished:
                            c = o.outputs[0]
                            self.gen_tokens += len(c.token_ids)
                            self.futs.pop(o.request_id).set_result((list(c.token_ids), c.finish_reason))
        except Exception:
            self.error = traceback.format_exc()
            raise
        finally:
            self.dead = True
            self.stop.set()
            if self.futs:
                self.eng.abort_request(list(self.futs))
            for f in self.futs.values():
                f.set_exception(Aborted())
            while not self.q.empty():
                self.q.get_nowait()[1].set_exception(Aborted())


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_done(path):
    """Finished (task_id, sample) pairs. A torn last line (killed mid-write) is cut off so appends stay well-formed."""
    done = set()
    if not path.exists():
        return done
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        path.write_bytes(raw)
    for ln in raw.decode().splitlines():
        r = json.loads(ln)
        done.add((r["task_id"], r["sample"]))
    return done


def check_resume(out, old, meta, cap_from):
    """Resume: refuse to mix settings in one traj.jsonl, with one documented exception (adds meta["cap_history"]).

    --resume-cap-from N allows only max_tokens N -> MAX_COMPLETION with N higher and nothing else changed. A lower cap
    acts only past MAX_COMPLETION tokens and the reward is frozen at the first submit, so a row at or under the new cap
    ends the same way, and a longer row keeps only a submit made inside the cap. The assert below needs every longer row
    at reward 0. The argument has one gap: a cut turn can parse differently. The old rows' end, truncated and length
    fields keep the old cap.
    """
    if "cap_history" in old:
        meta["cap_history"] = old["cap_history"]
    diff = {k: (old.get(k), v) for k, v in meta.items() if k not in ("versions", "env") and old.get(k) != v}
    if cap_from and cap_from > MAX_COMPLETION and set(diff) == {"sampling"}             and old["sampling"] == meta["sampling"].replace(f"max_tokens={MAX_COMPLETION},", f"max_tokens={cap_from},"):
        rows = [json.loads(ln) for ln in open(out / "traj.jsonl")]
        over = [r for r in rows if r["completion_tokens"] > MAX_COMPLETION]
        assert not any(r["reward"] for r in over), f"{sum(r['reward'] > 0 for r in over)} rows past the new cap scored: not reward-equivalent"
        meta["cap_history"] = meta.get("cap_history", []) + [
            {"rows": len(rows), "max_completion": cap_from, "then": MAX_COMPLETION, "rows_past_new_cap": len(over),
             "rewards_past_new_cap": 0, "why": "reward-equivalent: no row past the new cap scored"}]
        print(f"RESUME_EXCEPTION the first {len(rows)} rows ran at max_completion {cap_from}, the rest run at {MAX_COMPLETION}; "
              f"{len(over)} rows are past {MAX_COMPLETION}, none scored", flush=True)
        return
    assert not diff, f"settings differ from the run in {out}: {diff}"


def main():
    global MAX_COMPLETION
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # the unsloth/ repo training loads, e.g. unsloth/Qwen3-1.7B
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--decode", choices=DECODE)  # default resolved from the task splits below
    ap.add_argument("--max-completion", type=int, default=MAX_COMPLETION)  # must equal the trainer's completion cap
    ap.add_argument("--resume-cap-from", type=int)  # the cap the run's existing rows used, if higher (check_resume)
    ap.add_argument("--adapter")
    ap.add_argument("--util", type=float, default=0.84)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pool", default="~/llm-overseer-rl/tasks/v2")
    ap.add_argument("--allow-test", action="store_true")  # only for the one final test eval
    a = ap.parse_args()
    MAX_COMPLETION = a.max_completion  # read by Rollout and is_truncated

    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    pauses = [out / "PAUSE", HERE / "PAUSE"]
    if any(p.exists() for p in pauses):
        sys.exit(f"PAUSE file present ({[str(p) for p in pauses if p.exists()]}); remove it to run")
    pool = Path(os.path.expanduser(a.pool))
    codeenv.configure(pool, run_dir=out / "env")
    ids = [ln.strip() for ln in open(os.path.expanduser(a.tasks)) if ln.strip()]
    test = [t for t in ids if codeenv._tasks[t]["split"] == "test"]  # KeyError on an unknown id is intended
    assert not test or a.allow_test, f"{len(test)} test tasks in --tasks; test is touched only once, with --allow-test"
    if a.decode is None:  # val/test evals are greedy; any train task means calibration-style T 1.0
        a.decode = "train" if any(codeenv._tasks[t]["split"] == "train" for t in ids) else "greedy"

    tok, tmpl = make_tokenizer(a.model)
    eop = eos_and_pad(a.model, tok)
    import trl
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sp = SamplingParams(n=1, max_tokens=MAX_COMPLETION, detokenize=False, **DECODE[a.decode])
    meta = {"model": a.model, "adapter": a.adapter and os.path.abspath(os.path.expanduser(a.adapter)), "decode": a.decode,
            "sampling": repr(sp), "samples": a.samples, "tasks_file_sha256": sha_file(os.path.expanduser(a.tasks)),
            "pool_tasks_sha256": sha_file(pool / "tasks.jsonl"), "template_sha256": PINNED,
            "response_template_sha256": hashlib.sha256(json.dumps(tok.response_template, sort_keys=True, default=str).encode()).hexdigest(),
            "eos_and_pad": eop, "env": {"MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX")}, "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__}}
    mp = out / "meta.json"
    if mp.exists():
        check_resume(out, json.loads(mp.read_text()), meta, a.resume_cap_from)
    mp.write_text(json.dumps(meta, indent=1))

    tp = out / "traj.jsonl"
    done = load_done(tp)
    work = queue.Queue()
    for t in ids:
        for s in range(a.samples):
            if (t, s) not in done:
                work.put((t, s))  # task-major: a task's samples run together and share the prompt's prefix cache
    total = work.qsize()
    print(f"START {total} trajectories to run, {len(done)} already done, {len(ids)} tasks x {a.samples}", flush=True)
    if not total:
        print("EVAL nothing to do", flush=True)
        return 0

    t0 = time.perf_counter()
    llm = LLM(a.model, enable_lora=bool(a.adapter), max_lora_rank=16, max_loras=1, max_model_len=CTX,
              gpu_memory_utilization=a.util, seed=0, enable_prefix_caching=True)
    startup = time.perf_counter() - t0
    max_model_len = llm.llm_engine.model_config.max_model_len
    assert max_model_len == CTX
    lora = LoRARequest("adapter", 1, os.path.expanduser(a.adapter)) if a.adapter else None
    eng = Engine(llm, sp, lora)
    res = run_pool(eng, tok, tmpl, eop, max_model_len, work, out, pauses, a.workers)
    res["startup_secs"] = round(startup, 1)
    with open(out / "runs.jsonl", "a") as rf:  # one line per (re)start, for throughput across resumes
        rf.write(json.dumps({**res, "workers": a.workers, "util": a.util, "end": time.strftime("%F %T")}) + "\n")
    print("EVAL", json.dumps(res), flush=True)
    codeenv.sandbox().close()
    try:
        llm.llm_engine.engine_core.shutdown()
    except AttributeError:
        pass
    if eng.error:
        print("ENGINE Error", eng.error[-2000:], flush=True)
    return res["exit"]


def run_pool(eng, tok, tmpl, eop, max_model_len, work, out, pauses, workers):
    """Run every (task_id, sample) in the queue `work` on Engine `eng`, appending records to out/traj.jsonl.
    Shared by main() and eval/inproc_eval.py (the trainer's own engine). Polls `pauses` every 2 s. Returns stats
    with "exit": 0 done, 3 paused, 2 engine error or over 10 trajectory errors."""
    total = work.qsize()
    probe = codeenv.CodingEnv()
    tools = tool_methods(probe)  # the schema TRL renders comes from its probe instance (__init__ L662-679)
    lock, f = threading.Lock(), open(out / "traj.jsonl", "a")
    stats = {"done": 0, "errors": 0, "reward": 0.0}

    def worker():
        r = Rollout(copy.deepcopy(tok), tmpl, tools, eng.generate, max_model_len)  # own copy: Rust tokenizers share state
        env = codeenv.CodingEnv()
        while not eng.stop.is_set():
            try:
                t, s = work.get_nowait()
            except queue.Empty:
                return
            ts = time.time()
            try:
                o = r.run(env, t)
                reward = env.get_reward()
                rec = {"task_id": t, "base": codeenv._tasks[t]["base"], "sample": s, "reward": reward,
                       **describe(r._tokenizer, r, env, o, eop, max_model_len), "secs": round(time.time() - ts, 2)}
            except Aborted:
                return
            except Exception:  # noqa: BLE001 (logged and counted; the pair stays undone so a rerun retries it)
                with lock:
                    stats["errors"] += 1
                    with open(out / "errors.jsonl", "a") as ef:
                        ef.write(json.dumps({"task_id": t, "sample": s, "tb": traceback.format_exc()[-3000:]}) + "\n")
                    if stats["errors"] > 10:
                        eng.stop.set()
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                stats["done"] += 1
                stats["reward"] += reward

    gpu = threading.Thread(target=eng.loop, daemon=True)
    gpu.start()
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for w in ws:
        w.start()
    t1, last, paused = time.perf_counter(), 0.0, False
    while any(w.is_alive() for w in ws):
        time.sleep(2)
        if not paused and any(p.exists() for p in pauses):
            paused = True
            print("PAUSE seen: dropping in-flight trajectories and stopping", flush=True)
            eng.stop.set()
        el = time.perf_counter() - t1
        if el - last >= 60:
            last = el
            d = stats["done"]
            print(f"PROG {d}/{total} done, {el / 60:.1f} min, {eng.gen_tokens / el:.0f} gen tok/s, "
                  f"{3600 * d / el:.0f} traj/h, eta {(total - d) * el / max(d, 1) / 60:.0f} min, "
                  f"mean reward {stats['reward'] / max(d, 1):.3f}, in flight {len(eng.futs)}, errors {stats['errors']}",
                  flush=True)
    eng.stop.set()
    gpu.join(timeout=60)
    f.close()
    secs = time.perf_counter() - t1
    return {"done": stats["done"], "of": total, "errors": stats["errors"], "paused": paused, "wall_secs": round(secs, 1),
            "gen_tokens": eng.gen_tokens, "gen_tok_per_s": round(eng.gen_tokens / secs, 1),
            "prompt_tokens_submitted": eng.prompt_tokens, "requests": eng.requests,
            "traj_per_hour": round(3600 * stats["done"] / secs, 1), "mean_reward": round(stats["reward"] / max(stats["done"], 1), 4),
            "exit": 2 if eng.error or stats["errors"] > 10 else 3 if paused else 0}


if __name__ == "__main__":  # vLLM may spawn and re-import this file
    sys.exit(main())
