#!/usr/bin/env python3
"""Phase 0 benchmark and determinism runner for the patched teeworlds_srv.

Runs the server binaries built from the rl/phase0 branch with the sv_rl_* settings,
parses the rl_bench_done and rl_hash lines they print and writes one CSV row per run.

    run_bench.py matrix --release BUILD_DIR --bench BUILD_DIR --out results.csv
    run_bench.py determinism --bench BUILD_DIR --out determinism.txt

--release is a normal Release build, --bench is a Release build with -DCONF_DEBUG so
that dbg_dummies exists. Ports come from `freeport` and the servers bind to 127.0.0.1.
"""

import argparse
import csv
import datetime
import os
import re
import statistics
import subprocess
import sys
import tempfile

WARMUP_TICKS = 500  # matches RlBenchTick in server.cpp

DONE_RE = re.compile(
    r"rl_bench_done mode=(\d+) ticks=(\d+) wall_s=([\d.]+) cpu_s=([\d.]+) "
    r"tps=([\d.]+) us_per_tick=([\d.]+) catchups=(\d+)"
)
HASH_RE = re.compile(r"rl_hash tick=(\d+) size=(\d+) h=([0-9a-f]+)")


def freeport(n):
    out = subprocess.run(["freeport", "-n", str(n)], check=True, capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def release_port(port):
    subprocess.run(["freeport", "--release", str(port)], capture_output=True)


def server_args(port, mode, dummies, ticks, seed, hash_interval):
    cmds = [
        f"sv_port {port}",
        "bindaddr 127.0.0.1",
        "sv_register 0",
        "sv_auto_demo_record 0",
        "sv_map dm1",
        f"sv_rl_mode {mode}",
        f"sv_rl_seed {seed}",
        f"sv_rl_bench_ticks {ticks}",
        f"sv_rl_hash_interval {hash_interval}",
    ]
    if dummies:
        cmds.append(f"dbg_dummies {dummies}")
    return ["; ".join(cmds)]


def run_servers(build_dir, procs, mode, dummies, ticks, seed=1, hash_interval=0, timeout=900):
    """Start `procs` servers at once, wait for all, return one result dict per process."""
    ports = freeport(procs)
    binary = os.path.join(os.path.abspath(build_dir), "teeworlds_srv")
    running = []
    try:
        for port in ports:
            log = tempfile.TemporaryFile(mode="w+")
            p = subprocess.Popen(
                [binary] + server_args(port, mode, dummies, ticks, seed, hash_interval),
                cwd=build_dir, stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((p, log))
        results = []
        for p, log in running:
            # wait4 gives the peak RSS and CPU time of this one child
            _, status, usage = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            log.seek(0)
            text = log.read()
            log.close()
            m = DONE_RE.search(text)
            results.append({
                "exit": p.returncode,
                "done": m is not None,
                "mode": int(m.group(1)) if m else None,
                "ticks": int(m.group(2)) if m else None,
                "wall_s": float(m.group(3)) if m else None,
                "cpu_s": float(m.group(4)) if m else None,
                "tps": float(m.group(5)) if m else None,
                "us_per_tick": float(m.group(6)) if m else None,
                "catchups": int(m.group(7)) if m else None,
                "maxrss_kb": usage.ru_maxrss,
                "hashes": HASH_RE.findall(text),
                "log_tail": text[-2000:] if not m else "",
            })
        return results
    finally:
        for p, _ in running:
            if p.poll() is None:
                p.kill()
        for port in ports:
            release_port(port)


def machine_info():
    cpu = "unknown"
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return cpu, os.cpu_count()


def git_rev(path):
    r = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "")


# label, build key, mode, dummies, process counts
MATRIX = [
    ("A", "release", 0, 0, [1]),
    ("B", "release", 1, 0, [1]),
    ("C", "release", 2, 0, [1]),
    ("D0", "bench", 0, 2, [1]),
    ("D1", "bench", 1, 2, [1]),
    ("D2", "bench", 2, 2, [1]),
    ("E", "bench", 2, 2, [1, 2, 4, 8, 12]),
    ("F", "bench", 2, 0, [1]),
]


def cmd_matrix(a):
    builds = {"release": a.release, "bench": a.bench}
    cpu, ncpu = machine_info()
    rev = git_rev(a.release)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    fields = ["date", "teeworlds_rev", "cpu", "nproc", "label", "build", "mode", "dummies", "procs", "rep",
              "proc_index", "ticks", "wall_s", "cpu_s", "tps", "us_per_tick", "catchups", "maxrss_kb",
              "load1_before", "exit"]
    summary = []
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, build, mode, dummies, proc_list in MATRIX:
            if a.only and label not in a.only:
                continue
            ticks = a.ticks_realtime if mode == 0 else a.ticks_fast
            for procs in proc_list:
                totals, per_proc = [], []
                for rep in range(a.reps):
                    load1 = os.getloadavg()[0]
                    res = run_servers(builds[build], procs, mode, dummies, ticks)
                    for i, r in enumerate(res):
                        if not r["done"]:
                            print(f"{label} procs={procs} rep={rep} proc={i}: no rl_bench_done, exit={r['exit']}\n{r['log_tail']}", file=sys.stderr)
                        w.writerow({
                            "date": now, "teeworlds_rev": rev, "cpu": cpu, "nproc": ncpu, "label": label,
                            "build": build, "mode": mode, "dummies": dummies, "procs": procs, "rep": rep,
                            "proc_index": i, "ticks": r["ticks"], "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
                            "tps": r["tps"], "us_per_tick": r["us_per_tick"], "catchups": r["catchups"],
                            "maxrss_kb": r["maxrss_kb"], "load1_before": f"{load1:.2f}", "exit": r["exit"],
                        })
                    f.flush()
                    ok = [r for r in res if r["done"]]
                    totals.append(sum(r["tps"] for r in ok))
                    per_proc.extend(r["tps"] for r in ok)
                    print(f"{label} {build} mode={mode} dummies={dummies} procs={procs} rep={rep}: "
                          f"total_tps={totals[-1]:.0f} load1={load1:.2f} maxrss_kb={max(r['maxrss_kb'] for r in res)}",
                          flush=True)
                med = statistics.median(totals)
                spread = (max(totals) - min(totals)) / med if med else 0.0
                summary.append((label, mode, dummies, procs, med, statistics.median(per_proc), spread))
    print("\nlabel mode dummies procs median_total_tps median_per_proc_tps spread")
    for label, mode, dummies, procs, med, per, spread in summary:
        print(f"{label:5} {mode:4} {dummies:7} {procs:5} {med:16.0f} {per:19.0f} {spread:6.1%}")


def hash_seq(r):
    return [(int(t), h) for t, _, h in r["hashes"]]


def cmd_determinism(a):
    lines = []

    def report(name, ok, detail=""):
        line = f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip()
        lines.append(line)
        print(line, flush=True)

    def first_diff(x, y):
        for (tx, hx), (ty, hy) in zip(x, y):
            if tx != ty or hx != hy:
                return f"first difference at tick {tx}: {hx} vs {ty}: {hy}"
        if len(x) != len(y):
            return f"length {len(x)} vs {len(y)}"
        return ""

    hi = a.hash_interval
    # 1. same config twice
    r1 = run_servers(a.bench, 1, 2, 2, a.ticks_fast, seed=1, hash_interval=hi)[0]
    r2 = run_servers(a.bench, 1, 2, 2, a.ticks_fast, seed=1, hash_interval=hi)[0]
    s1, s2 = hash_seq(r1), hash_seq(r2)
    report("same_config_twice", s1 == s2 and len(s1) > 0, f"hashes={len(s1)} {first_diff(s1, s2)}")

    # 2. mode 0, 1 and 2 must give the same hashes for the realtime length
    by_mode = {}
    for mode in (0, 1, 2):
        r = run_servers(a.bench, 1, mode, 2, a.ticks_realtime, seed=1, hash_interval=hi)[0]
        by_mode[mode] = hash_seq(r)
        lines.append(f"     mode {mode}: hashes={len(by_mode[mode])} catchups={r['catchups']}")
    report("mode0_vs_mode1", by_mode[0] == by_mode[1] and len(by_mode[0]) > 0, first_diff(by_mode[0], by_mode[1]))
    report("mode1_vs_mode2", by_mode[1] == by_mode[2] and len(by_mode[1]) > 0, first_diff(by_mode[1], by_mode[2]))

    # 3. parallel processes match the single process run
    par = run_servers(a.bench, a.parallel, 2, 2, a.ticks_fast, seed=1, hash_interval=hi)
    bad = [i for i, r in enumerate(par) if hash_seq(r) != s1]
    report(f"parallel_{a.parallel}_vs_single", not bad, f"mismatching processes={bad}")

    # 4. other seed, informational only
    r3 = run_servers(a.bench, 1, 2, 2, a.ticks_fast, seed=2, hash_interval=hi)[0]
    same = hash_seq(r3) == s1
    lines.append(f"INFO seed_2_vs_seed_1 {'identical' if same else 'different'} {first_diff(s1, hash_seq(r3))}")
    print(lines[-1])

    with open(a.out, "w") as f:
        f.write(f"# {datetime.datetime.now().isoformat(timespec='seconds')} teeworlds {git_rev(a.bench)}\n")
        f.write("\n".join(lines) + "\n")
    if any(l.startswith("FAIL") for l in lines):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("matrix")
    m.add_argument("--release", required=True)
    m.add_argument("--bench", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--reps", type=int, default=3)
    m.add_argument("--ticks-fast", type=int, default=5000000)
    m.add_argument("--ticks-realtime", type=int, default=3000)
    m.add_argument("--only", nargs="*", help="run only these labels")
    m.set_defaults(func=cmd_matrix)

    d = sub.add_parser("determinism")
    d.add_argument("--bench", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--ticks-fast", type=int, default=1000000)
    d.add_argument("--ticks-realtime", type=int, default=3000)
    d.add_argument("--hash-interval", type=int, default=100)
    d.add_argument("--parallel", type=int, default=4)
    d.set_defaults(func=cmd_determinism)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
