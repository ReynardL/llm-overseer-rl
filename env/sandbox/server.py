"""Sandbox server. Started by client.Sandbox inside a delegated systemd --user scope, so each job gets its own
cgroup (memory, swap 0, pids, cpu) and can be killed as a whole with cgroup.kill. Runs on system Python, not a venv.

  /usr/bin/python3 -I -S server.py <socket path> <launcher.py>

Protocol: one JSON request line per connection, one JSON reply line.
  request: {"argv": [...], "binds": [[src, dst, writable]], "cwd": "/work", "timeout": s, "cap": bytes, "mem": bytes,
            "tmp": bytes, "env": {...}}
  reply:   {"exit": int, "timeout": bool, "oom": int, "secs": float, "stdout": str, "stderr": str, "truncated": bool}
Exits when its parent (the trainer or gate script) exits.
"""
import itertools
import json
import os
import selectors
import socket
import subprocess
import sys
import threading
import time

SOCK, LAUNCHER = sys.argv[1], sys.argv[2]
CG = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":", 2)[2]
JOB_MEM, JOB_PIDS, JOB_CPU = 384 << 20, 64, "100000 100000"  # 384 MiB RAM, no swap, 64 tasks, 1 CPU per job
_ids = itertools.count()
_slots = threading.Semaphore(8)


def setup_cgroups():
    os.makedirs(f"{CG}/srv", exist_ok=True)  # cgroup v2: no processes in a cgroup whose children use controllers
    with open(f"{CG}/srv/cgroup.procs", "w") as f:
        f.write(str(os.getpid()))
    with open(f"{CG}/cgroup.subtree_control", "w") as f:
        f.write("+memory +pids +cpu")


def wcg(path, name, val):
    with open(f"{path}/{name}", "w") as f:
        f.write(str(val))


def run_job(req):
    jcg = f"{CG}/j{next(_ids)}"
    os.mkdir(jcg)
    try:
        wcg(jcg, "memory.max", req.get("cg_mem", JOB_MEM))
        wcg(jcg, "memory.swap.max", 0)
        wcg(jcg, "pids.max", JOB_PIDS)
        wcg(jcg, "cpu.max", JOB_CPU)
        cfg = {k: req[k] for k in ("argv", "binds", "cwd", "timeout", "mem", "tmp", "nproc", "env") if k in req}
        cfg["cgroup"] = jcg
        cap, timeout = int(req.get("cap", 65536)), float(req.get("timeout", 10))
        t0 = time.perf_counter()
        p = subprocess.Popen(["/usr/bin/python3", "-I", "-S", LAUNCHER, json.dumps(cfg)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        bufs, truncated, killed = {p.stdout: bytearray(), p.stderr: bytearray()}, False, False
        sel = selectors.DefaultSelector()
        for fobj in bufs:
            sel.register(fobj, selectors.EVENT_READ)
        deadline = t0 + timeout + 2.0  # the launcher's own alarm fires at timeout; this is the backstop
        while sel.get_map():
            left = deadline - time.perf_counter()
            if left <= 0:
                killed = True
                break
            for key, _ in sel.select(min(left, 0.5)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                b = bufs[key.fileobj]
                b += chunk[: max(0, cap - len(b))]
                if len(b) >= cap:
                    truncated = True
            if truncated:  # huge output: stop the job instead of draining it
                break
        wcg(jcg, "cgroup.kill", 1)
        try:
            code = p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = -9
        secs = time.perf_counter() - t0
        oom = 0
        for ln in open(f"{jcg}/memory.events"):
            if ln.startswith("oom_kill "):
                oom = int(ln.split()[1])
        for fobj in bufs:
            fobj.close()
        return {"exit": code, "timeout": killed or code == 124, "oom": oom, "secs": round(secs, 4),
                "stdout": bufs[p.stdout].decode("utf-8", "replace"), "stderr": bufs[p.stderr].decode("utf-8", "replace"),
                "truncated": truncated}
    finally:
        for _ in range(50):  # rmdir succeeds once the kernel has reaped every task in the job cgroup
            try:
                os.rmdir(jcg)
                break
            except OSError:
                time.sleep(0.02)


def handle(conn):
    with conn:
        f = conn.makefile("rwb")
        try:
            req = json.loads(f.readline())
            with _slots:
                rep = run_job(req)
        except Exception as e:  # noqa: BLE001
            rep = {"error": repr(e)[:500]}
        f.write((json.dumps(rep) + "\n").encode())
        f.flush()


def main():
    setup_cgroups()
    parent = os.getppid()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    srv.bind(SOCK)
    os.umask(old)
    srv.listen(64)
    srv.settimeout(2.0)
    while os.getppid() == parent:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        conn.settimeout(None)
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
    os.unlink(SOCK)


main()
