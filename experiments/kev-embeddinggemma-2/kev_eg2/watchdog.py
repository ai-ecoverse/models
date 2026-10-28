"""Memory and time watchdog for one job on a shared Mac: runs the command in its own process group and kills the whole
group, writing the reason to its log, when any limit is breached.

    python -m kev_eg2.watchdog --log runs/arm/watchdog.log --heartbeat runs/arm/heartbeat.json -- <command...>

Every --interval seconds (default 3):
- job memory = RSS of the command and all its descendants (ps) + the MPS driver allocation the job reports in its
  heartbeat (torch.mps.driver_allocated_memory(), fresh within 120 s) > --max_gb (40);
- system: memory_pressure's free percentage < --min_free_pct (25), or the kernel's pressure level not normal
  (kern.memorystatus_vm_pressure_level != 1);
- swap used (vm.swapusage) more than --max_swap_rise_gb (2) above its value when the job started;
- wall clock past --deadline (epoch seconds; the experiment's 12-hour budget).
On a breach: SIGTERM to the group, SIGKILL after 10 s, exit 137. One job at a time: --lock is held (flock) for the
job's life and a second watchdog on the same lock refuses to start (exit 75).
"""
import argparse, fcntl, json, os, re, signal, subprocess, sys, time
from datetime import datetime


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def tree_rss_gb(root):
    kids, rss = {}, {}
    for line in sh("ps", "-A", "-o", "pid=,ppid=,rss=").splitlines():
        x = line.split()
        if len(x) == 3: kids.setdefault(int(x[1]), []).append(int(x[0])); rss[int(x[0])] = int(x[2])
    out, todo = [], [root]
    while todo:
        p = todo.pop(); out.append(p); todo += kids.get(p, [])
    return out, sum(rss.get(p, 0) for p in out) / 2**20


def free_pct():
    m = re.search(r"free percentage:\s*(\d+)%", sh("memory_pressure"))
    return int(m.group(1)) if m else None


def pressure_level():
    v = sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    return int(v) if v.isdigit() else None


def swap_used_gb():
    m = re.search(r"used = ([\d.]+)([MG])", sh("sysctl", "-n", "vm.swapusage"))
    return float(m.group(1)) / (1024 if m.group(2) == "M" else 1) if m else 0.0


def driver_gb(path):
    try:
        hb = json.load(open(path))
        if time.time() - hb["time"] < 120: return hb.get("mps_driver", 0) / 2**30
    except (OSError, ValueError, KeyError):
        pass
    return 0.0


def start_heartbeat(path, every=5.0):
    """For jobs without their own per-step heartbeat (scoring): a daemon thread reporting MPS bytes every few seconds."""
    import threading, torch
    def beat():
        while True:
            hb = {"pid": os.getpid(), "time": time.time()}
            if torch.backends.mps.is_available(): hb.update(mps_current=torch.mps.current_allocated_memory(), mps_driver=torch.mps.driver_allocated_memory())
            tmp = f"{path}.tmp"; open(tmp, "w").write(json.dumps(hb)); os.replace(tmp, path)
            time.sleep(every)
    threading.Thread(target=beat, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--heartbeat", default="")
    ap.add_argument("--lock", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", ".job.lock"))
    ap.add_argument("--max_gb", type=float, default=40); ap.add_argument("--min_free_pct", type=int, default=25)
    ap.add_argument("--max_swap_rise_gb", type=float, default=2); ap.add_argument("--interval", type=float, default=3)
    ap.add_argument("--deadline", type=float, default=0, help="epoch seconds; 0 = none")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(a.lock)), exist_ok=True)
    log = open(a.log, "a")
    def say(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} watchdog: {msg}"
        print(line, file=log, flush=True); print(line, file=sys.stderr, flush=True)
    lock = open(a.lock, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say(f"another job holds {a.lock}; refusing to start {' '.join(cmd[:4])}"); return 75
    if a.deadline and time.time() > a.deadline:
        say("the experiment's wall-clock budget is spent; not starting"); return 124
    swap0 = swap_used_gb()
    child = subprocess.Popen(cmd, start_new_session=True)
    lock.write(str(child.pid)); lock.flush()
    say(f"started pid {child.pid}: {' '.join(cmd)}; limits job {a.max_gb} GB, free >= {a.min_free_pct}%, pressure normal, "
        f"swap +{a.max_swap_rise_gb} GB (start {swap0:.2f} GB), deadline {datetime.fromtimestamp(a.deadline).isoformat(timespec='seconds') if a.deadline else 'none'}")
    peak = {"job_gb": 0.0, "rss_gb": 0.0, "driver_gb": 0.0, "min_free_pct": 100, "swap_gb": swap0}
    n, last_status, reason = 0, 0.0, None
    while child.poll() is None:
        _, rss = tree_rss_gb(child.pid); drv = driver_gb(a.heartbeat) if a.heartbeat else 0.0
        job = rss + drv
        free = free_pct() if n % 2 == 0 else peak.get("last_free", 100)   # memory_pressure takes ~0.5 s; every other poll
        level, swap = pressure_level(), swap_used_gb()
        peak.update(job_gb=max(peak["job_gb"], job), rss_gb=max(peak["rss_gb"], rss), driver_gb=max(peak["driver_gb"], drv),
                    min_free_pct=min(peak["min_free_pct"], free if free is not None else 100), swap_gb=max(peak["swap_gb"], swap), last_free=free)
        if job > a.max_gb: reason = f"job memory {job:.1f} GB (RSS {rss:.1f} + MPS driver {drv:.1f}) > {a.max_gb} GB"
        elif free is not None and free < a.min_free_pct: reason = f"system free memory {free}% < {a.min_free_pct}%"
        elif level is not None and level != 1: reason = f"system memory pressure level {level} (not normal)"
        elif swap - swap0 > a.max_swap_rise_gb: reason = f"swap used rose {swap - swap0:.2f} GB (> {a.max_swap_rise_gb} GB) since the start"
        elif a.deadline and time.time() > a.deadline: reason = "the experiment's 12-hour wall-clock budget is spent"
        if reason:
            say(f"KILL pid group {child.pid}: {reason}")
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            for _ in range(20):
                if child.poll() is not None: break
                time.sleep(0.5)
            pids, _ = tree_rss_gb(child.pid)
            for p in pids:
                try: os.kill(p, signal.SIGKILL)
                except ProcessLookupError: pass
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
            break
        if time.time() - last_status > 60:
            say(f"ok: job {job:.1f} GB (RSS {rss:.1f} + driver {drv:.1f}), free {free}%, level {level}, swap {swap:.2f} GB"); last_status = time.time()
        n += 1
        time.sleep(a.interval)
    peak.pop("last_free", None)
    say(f"exit {child.returncode}{' (killed: ' + reason + ')' if reason else ''}; peaks {json.dumps({k: round(v, 2) for k, v in peak.items()})}")
    with open(os.path.splitext(os.path.abspath(a.log))[0] + ".json", "w") as f:
        json.dump({"exit": child.returncode, "killed": reason, "peaks": peak, "cmd": cmd}, f, indent=1)
    return 137 if reason else child.returncode


if __name__ == "__main__":
    sys.exit(main())
