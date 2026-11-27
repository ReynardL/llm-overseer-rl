"""Client side of the sandbox: start the server once per process, then send it jobs.

    sb = Sandbox()                       # base dir defaults to $CRL/sbx (ext4, never /mnt/d)
    r = sb.run(["/usr/bin/python3", "-I", "-S", "/h/run_tests.py"], work="/path/to/ws", timeout=10)

Every job runs in the launcher's namespaces with the harness dir bound read-only at /h.
"""
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"


class Sandbox:
    def __init__(self, base=None):
        self.base = Path(base or os.path.expanduser(os.environ.get("CRL", "~/llm-overseer-rl")) + "/sbx")
        self.lib = self.base / f"lib-{os.getpid()}"
        self.sock = str(self.base / f"srv-{os.getpid()}.sock")
        self.proc = None
        self._start()

    def _start(self):
        # Copy launcher and harness to ext4: binds must not point into /mnt/d (9P), and the repo copy may change mid-run.
        if self.lib.exists():
            shutil.rmtree(self.lib)
        (self.lib / "h").mkdir(parents=True)
        shutil.copy(HERE / "launcher.py", self.lib / "launcher.py")
        shutil.copy(HERE / "server.py", self.lib / "server.py")
        for f in HARNESS.glob("*.py"):
            shutil.copy(f, self.lib / "h" / f.name)
        bus = f"/run/user/{os.getuid()}/bus"
        for _ in range(300):  # right after the distro boots, systemd --user takes a few seconds to open its bus
            if os.path.exists(bus):
                break
            time.sleep(0.05)
        env = {**os.environ, "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}"}
        # Scope-wide caps for all jobs together; per-job caps are set by the server.
        self.proc = subprocess.Popen(
            ["systemd-run", "--user", "--scope", "--quiet", f"--unit=crl-sbx-{os.getpid()}-{int(time.time())}",
             "-p", "Delegate=yes", "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", "-p", "TasksMax=1024",
             "-p", "CPUQuota=400%",
             "/usr/bin/python3", "-I", "-S", str(self.lib / "server.py"), self.sock, str(self.lib / "launcher.py")],
            env=env, stdin=subprocess.DEVNULL)
        for _ in range(200):
            if os.path.exists(self.sock):
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"sandbox server exited with {self.proc.returncode}")
            time.sleep(0.02)
        raise RuntimeError("sandbox server did not start")

    def run(self, argv, work=None, binds=(), timeout=10.0, cap=65536, **kw):
        """kw: mem (RLIMIT_AS bytes), tmp (tmpfs /tmp bytes), cg_mem, env, nproc, cwd."""
        b = [[str(self.lib / "h"), "/h", False]] + ([[str(work), "/work", False]] if work else []) + [list(x) for x in binds]
        req = {"argv": argv, "binds": b, "cwd": kw.pop("cwd", "/work" if work else "/"), "timeout": timeout, "cap": cap,
               "tmp": kw.pop("tmp", 1 << 20), **kw}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.sock)
            s.sendall((json.dumps(req) + "\n").encode())
            f = s.makefile("rb")
            rep = json.loads(f.readline())
        if "error" in rep:
            raise RuntimeError(f"sandbox server error: {rep['error']}")
        return rep

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        shutil.rmtree(self.lib, ignore_errors=True)
