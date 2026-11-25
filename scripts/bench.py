"""Run zsr and gplearn on the thirty Feynman equations with the same budget.

For each equation and seed: 1,000 training points and 1,000 test points drawn
from the equation's ranges, without noise. Each tool gets 30 seconds on one
thread (two runs at a time on the 2-vCPU machine), the same operators
(+ - * / sin cos exp log sqrt square) and 1,000 programs per generation.
gplearn keeps its other defaults, which fit mean absolute error, and stops
once that error is below 1e-10 of the target's standard deviation. zsr stops
30 generations after its first model with NMSE <= 1e-20.

A result counts as exact when SymPy finds it equal to the true formula after
each constant is rounded to a simple number (see common.is_exact), and its
predictions on the test points have NMSE <= 1e-12. The second condition
matters for gplearn, which prints constants to three decimals.

    python3 scripts/bench.py [--time 30] [--seeds 1,2] [--jobs 2]   # docs/data/bench.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
import warnings

import numpy as np
import sympy

from common import CACHE, EQUATIONS, N_TEST, N_TRAIN, dataset, is_exact, save_json, symbols, to_sympy, write_csv, zsr_binary
from equations import SPLIT

warnings.filterwarnings("ignore")
OPS = "add,sub,mul,div,sin,cos,exp,log,sqrt,square"
FIT_POINTS = 128  # zsr fits constants on this many points


def equation(ident):
    return next(e for e in EQUATIONS if e[0] == ident)


def truth_expr(eq):
    names = list(eq[3])
    return sympy.sympify(eq[2], locals=dict(zip(names, symbols(names)))), names


def predict(expr, names, x):
    f = sympy.lambdify(symbols(names), expr, "numpy")
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(f(*x.T), dtype=np.float64), (x.shape[0],))


def nmse(pred, y):
    with np.errstate(all="ignore"):
        v = float(np.mean((pred - y) ** 2) / np.var(y))
    return v if np.isfinite(v) else float("inf")


def run_zsr(eq, seed, budget):
    names, _, _ = dataset(eq, seed, N_TRAIN)
    path = CACHE / f"{eq[0]}-s{seed}.csv"
    out = CACHE / f"{eq[0]}-s{seed}.json"
    subprocess.run([str(zsr_binary()), str(path), "--time", str(budget), "--seed", str(seed), "--ops", OPS, "--json", str(out)],
                   check=True, capture_output=True)
    res = json.loads(out.read_text())
    best = res["front"][res["selected"]]
    found = to_sympy(best["expr"], names)
    point_evals = res["evaluations"] * N_TRAIN + res["lm_evaluations"] * min(FIT_POINTS, N_TRAIN)
    info = {"seconds": res["seconds"], "exact_fit_seconds": res["exact_at"], "size": best["size"],
            "generations": res["generations"], "point_evaluations": point_evals, "front": res["front"]}
    return found, (lambda x: predict(found, names, x)), info


def gplearn_functions():
    from gplearn.functions import make_function

    def _exp(a):
        with np.errstate(over="ignore"):
            return np.where(a < 100, np.exp(np.minimum(a, 100)), 0.0)

    def _square(a):
        with np.errstate(over="ignore"):
            return np.where(np.abs(a) < 1e100, a * a, 0.0)
    return ("add", "sub", "mul", "div", "sin", "cos", make_function(function=_exp, name="exp", arity=1), "log", "sqrt",
            make_function(function=_square, name="square", arity=1))


def gplearn_to_sympy(program, names):
    syms = symbols(names)
    local = {f"X{i}": s for i, s in enumerate(syms)}
    local.update({"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
                  "div": lambda a, b: a / b, "sqrt": lambda a: sympy.sqrt(abs(a)), "log": lambda a: sympy.log(abs(a)),
                  "sin": sympy.sin, "cos": sympy.cos, "exp": sympy.exp, "square": lambda a: a ** 2})
    return sympy.sympify(str(program), locals=local)


def run_gplearn(eq, seed, budget):
    from gplearn.genetic import SymbolicRegressor
    names, x, y = dataset(eq, seed, N_TRAIN)
    stop = 1e-10 * float(np.std(y))
    m = SymbolicRegressor(population_size=1000, generations=1, function_set=gplearn_functions(), random_state=seed,
                          n_jobs=1, warm_start=True, stopping_criteria=stop, verbose=0)
    t0 = time.time()
    gens = 0
    exact_t = None
    while time.time() - t0 < budget:
        gens += 1
        m.set_params(generations=gens)
        m.fit(x, y)
        if m._program.raw_fitness_ <= stop:
            exact_t = time.time() - t0
            break
    seconds = time.time() - t0
    found = gplearn_to_sympy(m._program, names)
    info = {"seconds": seconds, "exact_fit_seconds": exact_t, "size": int(m._program.length_), "generations": gens,
            "point_evaluations": gens * 1000 * N_TRAIN}
    return found, m.predict, info


def run(job):
    ident, seed, tool, budget = job
    eq = equation(ident)
    truth, names = truth_expr(eq)
    _, xt, yt = dataset(eq, seed + 1000, N_TEST)
    found, model, info = (run_zsr if tool == "zsr" else run_gplearn)(eq, seed, budget)
    test_nmse = nmse(model(xt), yt)
    return {"equation": ident, "split": SPLIT[ident], "seed": seed, "tool": tool, "expr": str(found),
            "exact": bool(test_nmse <= 1e-12 and is_exact(found, truth)), "test_nmse": test_nmse,
            "test_r2": 1 - test_nmse, **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=30)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated equation identifiers")
    ap.add_argument("--tools", default="zsr,gplearn")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    CACHE.mkdir(exist_ok=True)
    zsr_binary()
    seeds = [int(s) for s in args.seeds.split(",")]
    idents = [e[0] for e in EQUATIONS if not args.only or e[0] in args.only.split(",")]
    for ident in idents:
        eq = equation(ident)
        for seed in seeds:
            names, x, y = dataset(eq, seed, N_TRAIN)
            write_csv(CACHE / f"{ident}-s{seed}.csv", names, x, y, eq[1])
    jobs = [(ident, seed, tool, args.time) for ident in idents for seed in seeds for tool in args.tools.split(",")]
    rows = []
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "front"}), flush=True)
    save_json(args.out, {"time_budget": args.time, "seeds": seeds, "jobs": args.jobs, "runs": rows})


if __name__ == "__main__":
    main()
