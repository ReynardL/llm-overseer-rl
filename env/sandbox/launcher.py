"""Sandbox launcher: run one command in fresh namespaces with a minimal read-only root. No sudo, no bwrap.

Run as: /usr/bin/python3 -I -S launcher.py '<json config>'
Config: {"argv": [...], "binds": [[src, dst, writable], ...], "cwd": "/work", "timeout": s, "cgroup": path or null,
         "mem": bytes (RLIMIT_AS), "nproc": n, "fsize": bytes, "tmp": bytes (tmpfs /tmp size, 0 = none)}

Layers (gate 1 checks the configuration and resource limits):
- user ns: inside uid/gid 65534 mapped to the caller; exec as non-root clears every capability.
- mount ns + pivot_root into a tmpfs root holding only /usr (ro), /etc/ld.so.cache, /dev/null and the binds.
  /mnt (Windows drives), /run/WSL (interop sockets), /init (interop binfmt interpreter), $HOME and the venvs are absent.
- net ns (loopback only, down), pid ns (killing pid 1 kills everything), ipc, uts, cgroup ns.
- rlimits, no_new_privs, and a seccomp filter (no AF_VSOCK, no nested namespaces, no mount/ptrace/bpf/io_uring...).
- cgroup (memory, pids, cpu) when the server passes one.
Exit code: the command's, 128+signal if killed, 124 on timeout.
"""
import ctypes
import json
import os
import signal
import struct
import sys

cfg = json.loads(sys.argv[1])
libc = ctypes.CDLL(None, use_errno=True)


def chk(r, what):
    if r != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")


def mount(src, dst, fstype, flags, data=None):
    chk(libc.mount(src and src.encode(), dst.encode(), fstype and fstype.encode(), ctypes.c_ulong(flags),
                   data and data.encode()), f"mount {src} {dst}")


MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 4, 8, 32, 4096, 16384, 1 << 18
MS_NOATIME, MS_NODIRATIME, MS_RELATIME = 1024, 2048, 1 << 21


def ro_bind(src, dst, writable=False, noexec=False, dev=False):
    mount(src, dst, None, MS_BIND | MS_REC)  # a user ns refuses a non-recursive bind over locked submounts
    # Hide submounts (under /usr: /usr/lib/wsl/drivers is a 9p view of Windows files) behind empty read-only tmpfs.
    with open("/proc/self/mountinfo") as f:
        subs = sorted({ln.split()[4] for ln in f} - {src})
    subs = [m for m in subs if m.startswith(src.rstrip("/") + "/")]
    for m in subs:
        if not any(m.startswith(o + "/") for o in subs):
            mount("tmpfs", dst + m[len(src.rstrip("/")):], "tmpfs", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, "size=4k")
    # A remount in a user ns must keep the source mount's locked flags (nosuid, nodev, atime...).
    f = os.statvfs(src).f_flag
    keep = (f & (MS_NOSUID | MS_NODEV | MS_NOEXEC | MS_NOATIME | MS_NODIRATIME)) | (MS_RELATIME if f & 4096 else 0)
    mount(None, dst, None, MS_REMOUNT | MS_BIND | keep | MS_NOSUID | (0 if dev else MS_NODEV) | (0 if writable else MS_RDONLY)
          | (MS_NOEXEC if noexec else 0))


def seccomp():
    # x86_64 syscall numbers. EPERM for each; clone3 gets ENOSYS so glibc falls back to clone (flags checked below).
    deny = [101, 165, 166, 155, 167, 168, 169, 272, 308, 425, 426, 427, 321, 298, 323, 250, 248, 249, 246, 320, 175,
            313, 176, 310, 311, 304, 303, 428, 429, 430, 431, 432, 433, 442, 163, 179, 172, 173, 153, 103, 134, 139]
    # 101 ptrace, 165/166 mount/umount2, 155 pivot_root, 167-169 swap/reboot, 272 unshare, 308 setns, 425-427 io_uring,
    # 321 bpf, 298 perf_event_open, 323 userfaultfd, 248-250 keys, 246/320 kexec, 175/313/176 modules,
    # 310/311 process_vm_*, 303/304 handles, 428-433/442 new mount API, 163 acct, 179 quotactl, 172/173 io ports,
    # 153 vhangup, 103 syslog, 134 uselib, 139 sysfs
    LD, JEQ, RET, AND = 0x20, 0x15, 0x06, 0x54
    ALLOW, KILL = 0x7FFF0000, 0x80000000
    EPERM, ENOSYS = 0x00050000 | 1, 0x00050000 | 38
    NS_FLAGS = 0x10000000 | 0x00020000 | 0x40000000 | 0x20000000 | 0x08000000 | 0x04000000 | 0x02000000  # user mnt net pid ipc uts cgroup
    p = [(LD, 0, 0, 4), (JEQ, 1, 0, 0xC000003E), (RET, 0, 0, KILL), (LD, 0, 0, 0),
         (0x35, 0, 1, 0x40000000), (RET, 0, 0, KILL)]  # JGE x32 ABI -> kill
    for nr in deny:
        p += [(JEQ, 0, 1, nr), (RET, 0, 0, EPERM)]
    p += [(JEQ, 0, 1, 435), (RET, 0, 0, ENOSYS)]
    p += [(JEQ, 0, 3, 41), (LD, 0, 0, 16), (JEQ, 0, 1, 40), (RET, 0, 0, EPERM)]  # socket(AF_VSOCK)
    p += [(LD, 0, 0, 0), (JEQ, 0, 4, 56), (LD, 0, 0, 16), (AND, 0, 0, NS_FLAGS), (JEQ, 1, 0, 0), (RET, 0, 0, EPERM)]  # clone
    p += [(RET, 0, 0, ALLOW)]
    buf = b"".join(struct.pack("HBBI", *ins) for ins in p)
    prog = struct.pack("HxxxxxxP", len(p), ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value)
    chk(libc.prctl(38, 1, 0, 0, 0), "no_new_privs")
    pbuf = ctypes.create_string_buffer(prog)
    chk(libc.prctl(22, 2, ctypes.c_void_p(ctypes.addressof(pbuf)), 0, 0), "seccomp")
    return buf  # keep alive until exec


def main():
    if cfg.get("cgroup"):
        with open(os.path.join(cfg["cgroup"], "cgroup.procs"), "w") as f:
            f.write(str(os.getpid()))
    uid, gid = os.getuid(), os.getgid()
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS | os.CLONE_NEWNET | os.CLONE_NEWPID | os.CLONE_NEWIPC
               | os.CLONE_NEWUTS | os.CLONE_NEWCGROUP)
    for name, text in (("setgroups", "deny"), ("uid_map", f"65534 {uid} 1"), ("gid_map", f"65534 {gid} 1")):
        with open(f"/proc/self/{name}", "w") as f:
            f.write(text)
    mount(None, "/", None, MS_REC | MS_PRIVATE)
    r = "/mnt"  # tmpfs over /mnt in this mount ns only; it becomes the new root
    mount("tmpfs", r, "tmpfs", MS_NOSUID | MS_NODEV, "size=1m,nr_inodes=256,mode=755")
    for d in ("usr", "etc", "dev", "tmp", ".old"):
        os.mkdir(f"{r}/{d}")
    for link in ("bin", "lib", "lib64", "sbin"):
        os.symlink(f"usr/{link}", f"{r}/{link}")
    ro_bind("/usr", f"{r}/usr")
    for fpath in ("/etc/ld.so.cache", "/dev/null"):
        open(f"{r}{fpath}", "w").close()
        ro_bind(fpath, f"{r}{fpath}", writable=fpath == "/dev/null", dev=fpath == "/dev/null")
    for src, dst, writable in cfg.get("binds", []):
        os.makedirs(f"{r}{dst}", exist_ok=True)
        ro_bind(src, f"{r}{dst}", writable, noexec=True)
    if cfg.get("tmp"):
        mount("tmpfs", f"{r}/tmp", "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, f"size={cfg['tmp']},nr_inodes=512,mode=1777")
    os.chdir(r)
    chk(libc.syscall(155, b".", b".old"), "pivot_root")
    os.chdir("/")
    chk(libc.umount2(b"/.old", 2), "umount old root")  # MNT_DETACH
    os.rmdir("/.old")
    mount(None, "/", None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV)

    pid = os.fork()
    if pid == 0:  # pid 1 of the new pid ns
        try:
            import resource as R
            lim = [(R.RLIMIT_AS, cfg.get("mem", 1 << 30)), (R.RLIMIT_CPU, int(cfg.get("timeout", 10)) + 1),
                   (R.RLIMIT_FSIZE, cfg.get("fsize", 8 << 20)), (R.RLIMIT_NOFILE, 64), (R.RLIMIT_NPROC, cfg.get("nproc", 16)),
                   (R.RLIMIT_CORE, 0), (R.RLIMIT_MEMLOCK, 0), (R.RLIMIT_MSGQUEUE, 0)]
            for k, v in lim:
                R.setrlimit(k, (v, v))
            os.closerange(3, 1 << 16)
            os.chdir(cfg.get("cwd", "/"))
            keep = seccomp()  # noqa: F841
            env = {"PATH": "/usr/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
                   "HOME": "/tmp", **cfg.get("env", {})}
            os.execve(cfg["argv"][0], cfg["argv"], env)
        except BaseException as e:
            os.write(2, f"SANDBOX_SETUP_ERROR {e!r}\n".encode())
            os._exit(125)
    timed_out = []

    def on_alarm(*_):
        timed_out.append(1)
        os.kill(pid, signal.SIGKILL)  # pid 1 dies, the kernel kills the rest of the pid ns

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(cfg.get("timeout", 10)))
    while True:
        try:
            _, st = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
    signal.setitimer(signal.ITIMER_REAL, 0)
    os._exit(124 if timed_out else (os.WEXITSTATUS(st) if os.WIFEXITED(st) else 128 + os.WTERMSIG(st)))


main()
