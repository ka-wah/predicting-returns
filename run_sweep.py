#!/usr/bin/env python3
"""
run_sweep.py — launch a grid of runs by cloning a base YAML and overriding fields.

Defaults to: python main.py -c {cfg}
Placeholders supported in --main-cmd:
  {cfg} -> path to generated override YAML
  {tag} -> run tag like "call-atm-20250101-120000-0003"

Examples:
  python run_sweep.py --base conf.yaml --callput all,call,put --moneyness all,atm,otm,itm \
    --cp-key data.callput --mny-key data.moneyness --parallel 4 --keep-going \
    --set training.seed=42 --env WANDB_MODE=offline
"""
from __future__ import annotations

import argparse
import itertools as it
import os, shlex, signal, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

TOTAL_START = time.time()

try:
    import yaml
except Exception:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise

# -----------------------
# Small utilities
# -----------------------

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def yaml_deepcopy(obj):
    # deepcopy is fine for standard types; yaml roundtrip is slower.
    return deepcopy(obj)

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def deep_set(d: dict, path: List[str], value):
    """Set d[path[0]]...[path[-1]] = value, creating dicts as needed."""
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value

def try_set_many(d: dict, candidates: List[List[str]], value) -> Optional[List[str]]:
    """Try several keypaths; return the one that succeeded, else None."""
    for path in candidates:
        try:
            deep_set(d, path, value)
            return path
        except Exception:
            continue
    return None

def parse_kv(s: str) -> Tuple[str, object]:
    """
    Parse "a.b.c=val" -> ("a.b.c", typed_val). Value is parsed with YAML to get types.
    """
    if "=" not in s:
        raise ValueError(f"--set/--env expects key=value, got: {s}")
    k, v = s.split("=", 1)
    k = k.strip()
    # For env we keep it as string; for set we YAML-parse below
    return k, v

def format_tag(cp: str, mny: str, idx: int, stamp: Optional[str] = None) -> str:
    # timestamp + 4-digit index improves stability for parallel runs
    ts = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{cp}-{mny}-{ts}-{idx:04d}"

def run_one(cmd_list: List[str], log_path: Path, retries: int = 0, delay_between_retries: float = 1.0) -> Tuple[int, float]:
    start = time.time()
    attempt = 0
    rc = -1
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while attempt <= retries:
        with log_path.open("w", encoding="utf-8") as logf:
            try:
                proc = __import__("subprocess").Popen(cmd_list, stdout=logf, stderr=__import__("subprocess").STDOUT)
                rc = proc.wait()
            except KeyboardInterrupt:
                # Let main handle coordinated shutdown; still close the file and propagate
                raise
        if rc == 0 or attempt == retries:
            break
        attempt += 1
        time.sleep(delay_between_retries)
    elapsed = time.time() - start
    return rc, elapsed

# -----------------------
# Override builder
# -----------------------

def build_override(
    base_cfg: dict,
    callput: str,
    moneyness: str,
    out_tag: str,
    cp_key: Optional[str],
    mny_key: Optional[str],
    outdir_key: Optional[str],
    out_root: Optional[Path],
    target_scale_kind: Optional[str],
    extra_sets: List[Tuple[str, object]],
) -> dict:
    cfg = yaml_deepcopy(base_cfg)

    # Candidate paths across common layouts
    cp_candidates = []
    if cp_key:
        cp_candidates.append(cp_key.split("."))
    cp_candidates += [
        ["data", "callput"],
        ["data", "side"],
        ["data", "filters", "callput"],
        ["dataset", "filters", "callput"],
        ["filters", "callput"],
        ["selection", "callput"],
    ]
    mny_candidates = []
    if mny_key:
        mny_candidates.append(mny_key.split("."))
    mny_candidates += [
        ["data", "moneyness"],
        ["data", "filters", "moneyness"],
        ["dataset", "filters", "moneyness"],
        ["filters", "moneyness"],
        ["selection", "moneyness"],
    ]

    # Apply call/put & moneyness
    set_cp = try_set_many(cfg, cp_candidates, callput)
    set_mn = try_set_many(cfg, mny_candidates, moneyness)

    if target_scale_kind:
        deep_set(cfg, ["data", "target_scale", "kind"], target_scale_kind)

    # Optional per-run output directory
    if outdir_key and out_root:
        deep_set(cfg, outdir_key.split("."), str(out_root / out_tag))

    # Stash run tag
    deep_set(cfg, ["run_tag"], out_tag)

    # Arbitrary --set overrides
    for key, value in extra_sets:
        deep_set(cfg, key.split("."), value)

    return cfg

# -----------------------
# Main
# -----------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a grid of configs from a base YAML.")
    parser.add_argument("--base", required=True, help="Path to base YAML to clone/override.")
    parser.add_argument("--main-cmd", default="python main.py -c {cfg}",
                        help="Entrypoint template. Use {cfg} and optionally {tag}.")
    parser.add_argument("--callput", default="all,call,put", help="Comma-separated values for call/put side.")
    parser.add_argument("--moneyness", default="all,atm,otm,itm", help="Comma-separated moneyness buckets.")
    parser.add_argument("--cp-key", default="", help="Dot-path to callput key in YAML (e.g., 'filters.callput').")
    parser.add_argument("--mny-key", default="", help="Dot-path to moneyness key in YAML (e.g., 'filters.moneyness').")
    parser.add_argument("--target-scale-kind", default="", help="Override target_scale.kind in YAML.")
    parser.add_argument("--outdir-key", default="", help="Optional dot-path to per-run output dir key (e.g., 'out.dir').")
    parser.add_argument("--out-root", default="results/experiments/sweeps",
                        help="Root folder for per-run outputs if --outdir-key is set.")
    parser.add_argument("--sweep-dir", default="sweeps", help="Where to write override cfgs and logs.")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands; do not execute.")
    parser.add_argument("--keep-going", action="store_true", help="Keep running even if a job fails.")
    parser.add_argument("--retry", type=int, default=0, help="Retry failed runs this many times.")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds to wait between retries.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between starting jobs (serial or parallel queueing).")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        help="Arbitrary overrides: key=val (YAML-typed). Can be used multiple times.")
    parser.add_argument("--env", dest="envs", action="append", default=[],
                        help="Environment variables for the child process: KEY=VAL. Can be used multiple times.")
    args = parser.parse_args(argv)

    base_path = Path(args.base).expanduser().resolve()
    if not base_path.exists():
        raise SystemExit(f"[error] Base config not found: {base_path}")

    base_cfg = load_yaml(base_path)

    # Validate placeholders
    if "{cfg}" not in args.main_cmd:
        print("[warn] --main-cmd does not contain {cfg}; are you sure?", file=sys.stderr)

    # Parse lists
    callputs = parse_csv_list(args.callput)
    moneynesses = parse_csv_list(args.moneyness)

    # Directories
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve() if args.outdir_key else None
    out_root_name = out_root.name if out_root else "default"

    cfg_base_dir = sweep_dir / out_root_name / "cfg"
    log_base_dir = sweep_dir / out_root_name / "log"
    cfg_base_dir.mkdir(parents=True, exist_ok=True)
    log_base_dir.mkdir(parents=True, exist_ok=True)

    # Extra overrides & env
    extra_sets: List[Tuple[str, object]] = []
    for item in args.sets:
        k, raw_v = parse_kv(item)
        # YAML-parse values to get ints/floats/bools/lists/etc.
        try:
            v = yaml.safe_load(raw_v)
        except Exception as e:
            raise SystemExit(f"[error] Could not parse value for --set {item!r}: {e}")
        extra_sets.append((k, v))

    child_env = os.environ.copy()
    for item in args.envs:
        k, v = parse_kv(item)
        child_env[str(k)] = str(v)

    # Build job list
    combos = list(it.product(callputs, moneynesses))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs: List[Tuple[List[str], Path]] = []

    for idx, (cp, mny) in enumerate(combos, start=1):
        tag = format_tag(cp=cp, mny=mny, idx=idx, stamp=stamp)
        cfg = build_override(
            base_cfg=base_cfg,
            callput=cp,
            moneyness=mny,
            out_tag=tag,
            cp_key=args.cp_key or None,
            mny_key=args.mny_key or None,
            outdir_key=args.outdir_key or None,
            out_root=out_root,
            target_scale_kind=(args.target_scale_kind or None),
            extra_sets=extra_sets,
        )

        out_cfg_path = cfg_base_dir / f"{tag}.yaml"
        save_yaml(out_cfg_path, cfg)

        main_cmd = args.main_cmd.format(cfg=str(out_cfg_path), tag=tag)
        # On Windows, posix=False is correct; else True.
        cmd_list = shlex.split(main_cmd, posix=(os.name != "nt"))

        if args.dry_run:
            print(f"[dry-run] {tag} :: {' '.join(cmd_list)}")
        else:
            log_path = log_base_dir / f"{tag}.log"
            jobs.append((cmd_list, log_path))
        if args.delay > 0 and args.parallel <= 1 and not args.dry_run:
            time.sleep(args.delay)

    if args.dry_run:
        return

    # Graceful shutdown: propagate SIGINT to child threads nicely
    interrupted = {"flag": False}

    def _sigint_handler(signum, frame):
        if not interrupted["flag"]:
            interrupted["flag"] = True
            print("\n[info] Caught Ctrl-C. Finishing current tasks; no new tasks will start.", file=sys.stderr)
        else:
            print("[info] Second Ctrl-C. Exiting now.", file=sys.stderr)
            os._exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Execute jobs
    started = 0
    failed = 0
    t0 = time.time()

    if args.parallel <= 1:
        for cmd, log in jobs:
            if interrupted["flag"]:
                break
            started += 1
            print(f"[run]  {' '.join(cmd)}  -> {log}")
            rc, elapsed = run_one(cmd, log, retries=args.retry, delay_between_retries=args.retry_wait)
            status = "OK" if rc == 0 else f"RC={rc}"
            print(f"[done] {' '.join(cmd)} -> {log}  [{status}] [time: {elapsed:.2f}s]")
            if rc != 0:
                failed += 1
                tag = log.stem
                parts = tag.split("-")
                side = parts[0] if parts else "?"
                money = parts[1] if len(parts) > 1 else "?"
                print(f"[error] run failed (side={side}, moneyness={money}).", file=sys.stderr)
                if not args.keep_going:
                    sys.exit(rc)
            if args.delay > 0:
                time.sleep(args.delay)
    else:
        print(f"[info] launching up to {args.parallel} concurrent runs")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {}
            for cmd, log in jobs:
                if interrupted["flag"]:
                    break
                # Optionally stagger submissions
                if args.delay > 0 and started > 0:
                    time.sleep(args.delay)
                fut = ex.submit(run_one, cmd, log, args.retry, args.retry_wait)
                futures[fut] = (cmd, log)
                started += 1

            for fut in as_completed(futures):
                cmd, log = futures[fut]
                try:
                    rc, elapsed = fut.result()
                except KeyboardInterrupt:
                    interrupted["flag"] = True
                    break
                ok = "OK" if rc == 0 else f"RC={rc}"
                print(f"[done] {' '.join(cmd)} -> {log}  [{ok}] [time: {elapsed:.2f}s]")
                if rc != 0:
                    failed += 1
                    tag = log.stem
                    parts = tag.split("-")
                    side = parts[0] if parts else "?"
                    money = parts[1] if len(parts) > 1 else "?"
                    print(f"[error] failure (side={side}, moneyness={money}).", file=sys.stderr)
                    if not args.keep_going:
                        sys.exit(rc)

    total_elapsed = time.time() - t0
    print(f"\n[summary] runs started: {started} | failures: {failed} | elapsed: {total_elapsed:.2f}s")
    print(f"[♥] Done! Total wall time: {time.time() - TOTAL_START:.2f}s")
    if failed and not args.keep_going:
        sys.exit(1)


if __name__ == "__main__":
    main()
