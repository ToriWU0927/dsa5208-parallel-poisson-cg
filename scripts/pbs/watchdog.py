#!/usr/bin/env python3
"""Job watchdog for Vanda/Atlas: stop jobs that burn core-hours
without doing useful work. Runs on the LOGIN node, polls this user's jobs and
deletes (qdel) a job when

  1. start-failure loop: run_count >= 2 (PBS keeps re-starting a job that fails
     to launch, Exit_status -3, ~20 times before holding it, and the attempts
     use credits), or held by the system after failed attempts;
  2. stall: the job is running but its output file has not changed for
     --stall seconds (scripts print a time-stamped line per configuration,
     and every configuration has its own shorter timeout inside the job,
     so silence for this long means the job itself is stuck);
  3. dependency deadlock: held on a dependency whose parent no longer exists
     or failed.

Every action is printed with a time stamp and appended to ~/dsa5208/watchdog.log.
Exits when the user has no jobs left (or after --max-hours).

  python3 scripts/pbs/watchdog.py [--stall 900] [--interval 30] [--dry-run]
"""
import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path.home() / "dsa5208" / "watchdog.log"


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode, r.stdout


def my_jobs(user):
    """Returns (ids, details). ids is None if qstat itself failed (then keep waiting)."""
    rc, out = sh(["qstat", "-w", "-u", user])  # -w: untruncated job ids
    if rc != 0 and not out.strip():
        return None, {}
    ids = [ln.split()[0] for ln in out.splitlines() if ln[:1].isdigit()]
    details = {}
    for jid in ids:  # one call per job: a job finishing in between must not hide the others
        rc, out = sh(["qstat", "-f", "-F", "json", jid])
        if out.strip():
            try:
                details.update(json.loads(out).get("Jobs", {}))
            except json.JSONDecodeError:
                pass
    return ids, details


def output_file(job):
    path = job.get("Output_Path", "")
    return Path(path.split(":", 1)[-1]) if path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall", type=int, default=900, help="seconds without output -> qdel")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=12)
    ap.add_argument("--dry-run", action="store_true", help="report only, never qdel")
    args = ap.parse_args()
    user = getpass.getuser()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"watchdog started (stall {args.stall}s, interval {args.interval}s"
        f"{', dry run' if args.dry_run else ''})")
    t_end = time.time() + args.max_hours * 3600
    seen_start = {}

    def kill(jid, why):
        log(f"QDEL {jid}: {why}")
        if not args.dry_run:
            sh(["qdel", jid])

    while time.time() < t_end:
        ids, jobs = my_jobs(user)
        if ids is None:  # qstat unavailable for a moment
            time.sleep(args.interval)
            continue
        if not ids:
            log("no jobs left, watchdog exits")
            return 0
        for jid, j in jobs.items():
            name = j.get("Job_Name", "?")
            state = j.get("job_state", "?")
            runs = int(j.get("run_count", 0))
            comment = j.get("comment", "")
            tag = f"{jid} ({name})"
            if runs >= 2 or "too many failed attempts" in comment:
                kill(tag, f"start-failure loop (run_count={runs}, Exit_status="
                          f"{j.get('Exit_status', '?')}, comment: {comment[:80]})")
                continue
            if state == "H" and "depend" in j and "afterok" in j["depend"]:
                parents = [p.split("@")[0] for p in j["depend"].split(":")[1:]]
                for p in parents:
                    rc, out = sh(["qstat", "-x", "-f", "-F", "json", p])
                    try:
                        pj = json.loads(out).get("Jobs", {}).get(p, {}) if rc == 0 and out.strip() else {}
                    except json.JSONDecodeError:  # invalid qstat output: check again next round
                        pj = {}
                    if pj.get("job_state") == "F" and int(pj.get("Exit_status", 0)) != 0:
                        kill(tag, f"dependency {p} finished with Exit_status {pj.get('Exit_status')}")
                        break
                continue
            if state == "R":
                seen_start.setdefault(jid, time.time())
                f = output_file(j)
                last = f.stat().st_mtime if f and f.exists() else seen_start[jid]
                idle = time.time() - max(last, seen_start[jid])
                if idle > args.stall:
                    kill(tag, f"no output for {idle:.0f}s (> {args.stall}s) -> stuck")
        time.sleep(args.interval)
    log("max-hours reached, watchdog exits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
