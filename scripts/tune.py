"""How the defaults were chosen: zsr under different settings on the
development equations it found hard, with several search seeds each and the
benchmark's time budget, two runs at a time. "defaults" are the settings
before tuning: linear scaling, migration every 5 generations, parsimony 0.12.
The held-out equations are never used here.

    python3 scripts/tune.py --round 1 [--time 30] [--seeds 101,102,103] [--jobs 2]
    python3 scripts/tune.py --round 2                # both into docs/data/tuning.json
"""
import argparse
import concurrent.futures as cf
import json
import subprocess

from bench import OPS, truth_expr
from common import CACHE, N_TRAIN, RESULTS, dataset, is_exact, save_json, to_sympy, write_csv, zsr_binary
from equations import DEVELOPMENT

ROUNDS = {
    # Round 1: one change at a time from the settings before tuning.
    "1": [
        ("defaults", ["--parsimony", "0.12", "--migrate-every", "5", "--insert", "0"]),
        ("no linear scaling", ["--scaling", "off", "--parsimony", "0.12", "--migrate-every", "5", "--insert", "0"]),
        ("migration every 50 generations", ["--migrate-every", "50", "--parsimony", "0.12", "--insert", "0"]),
        ("parsimony 0.03", ["--parsimony", "0.03", "--migrate-every", "5", "--insert", "0"]),
        ("migration every 50, parsimony 0.03", ["--migrate-every", "50", "--parsimony", "0.03", "--insert", "0"]),
        ("no scaling, migration every 50, parsimony 0.03",
         ["--scaling", "off", "--migrate-every", "50", "--parsimony", "0.03", "--insert", "0"]),
    ],
    # Round 2, from round 1's best settings (re-run as a control, after
    # crossover was changed to never fail on an intermediate count of
    # constants): the insertion mutation, and a different parsimony on each
    # island.
    "2": [
        ("migration every 50, parsimony 0.03 (control)", ["--migrate-every", "50", "--parsimony", "0.03", "--insert", "0"]),
        ("... with insertion 0.1", ["--migrate-every", "50", "--parsimony", "0.03", "--insert", "0.1"]),
        ("... with parsimony 0.01/0.03/0.06/0.12 by island",
         ["--migrate-every", "50", "--parsimony", "0.01,0.03,0.06,0.12", "--insert", "0"]),
        ("... with both", ["--migrate-every", "50", "--parsimony", "0.01,0.03,0.06,0.12", "--insert", "0.1"]),
    ],
}
HARD = ["I.8.14", "I.10.7", "I.13.4", "I.15.10", "I.18.4", "I.27.6", "I.37.4"]


def run(job):
    config, args, ident, seed, budget = job
    eq = next(e for e in DEVELOPMENT if e[0] == ident)
    names = list(eq[3])
    path = CACHE / f"{ident}-s1.csv"
    out = CACHE / f"tune-{ident}-{seed}-{abs(hash(config))}.json"
    subprocess.run([str(zsr_binary()), str(path), "--time", str(budget), "--seed", str(seed), "--ops", OPS,
                    "--json", str(out)] + args, check=True, capture_output=True)
    res = json.loads(out.read_text())
    best = res["front"][res["selected"]]
    truth, _ = truth_expr(eq)
    found = to_sympy(best["expr"], names)
    exact = best["nmse"] <= 1e-12 and is_exact(found, truth)  # SymPy only for numerically exact models
    return {"config": config, "equation": ident, "seed": seed, "exact": bool(exact),
            "exact_fit_seconds": res["exact_at"], "nmse": best["nmse"], "size": best["size"], "expr": str(found)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=30)
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--round", default="1", choices=sorted(ROUNDS))
    ap.add_argument("--configs", default=None, help="indices of the round's configurations to run (default all)")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    zsr_binary()
    for ident in HARD:
        eq = next(e for e in DEVELOPMENT if e[0] == ident)
        names, x, y = dataset(eq, 1, N_TRAIN)
        write_csv(CACHE / f"{ident}-s1.csv", names, x, y, eq[1])
    configs = ROUNDS[args.round]
    if args.configs:
        configs = [configs[int(i)] for i in args.configs.split(",")]
    jobs = [(name, a, ident, int(s), args.time) for name, a in configs for ident in HARD for s in args.seeds.split(",")]
    rows = []
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {name: sum(r["exact"] for r in rows if r["config"] == name) for name, _ in configs}
    print(json.dumps(summary, indent=1))
    path = RESULTS / "tuning.json"
    data = json.loads(path.read_text()) if path.exists() else {"time_budget": args.time, "equations": HARD, "rounds": {}}
    old = data["rounds"].get(args.round)
    if old and args.configs:  # replace only the configurations run now
        names = {n for n, _ in configs}
        rows = [r for r in old["runs"] if r["config"] not in names] + rows
        summary = {**old["summary"], **summary}
        summary = {n: summary[n] for n, _ in ROUNDS[args.round]}
    data["rounds"][args.round] = {"seeds": args.seeds, "configs": [{"name": n, "args": a} for n, a in ROUNDS[args.round]],
                                  "summary": summary, "runs": rows}
    save_json("tuning.json", data)

if __name__ == "__main__":
    main()
