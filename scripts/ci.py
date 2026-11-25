"""The checks that run in CI, after the unit tests.

Builds zsr with safety checks on (ReleaseSafe) and runs it for a fixed number
of generations, which makes a run deterministic, on equations it should
solve. Each run must
  1. report a model that SymPy confirms is the true law (common.is_exact);
  2. report a Pareto front whose NMSE falls as size grows;
  3. give the same front when run again with the same seed.

    ZIG="python3 -m ziglang" python3 scripts/ci.py
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import sympy

from common import CACHE, EQUATIONS, ROOT, dataset, is_exact, to_sympy, write_csv

ZIG = shlex.split(os.environ.get("ZIG", "zig"))
OPS = "add,sub,mul,div,sin,cos,exp,log,sqrt,square"
CASES = [("I.12.1", 1), ("I.14.4", 1), ("I.12.4", 2), ("I.39.10", 1), ("I.34.8", 2), ("I.6.2a", 2), ("I.24.6", 4),
         ("II.11.3", 3)]
GENERATIONS = 300


def main():
    subprocess.run(ZIG + ["build", "-Doptimize=ReleaseSafe"], cwd=ROOT, check=True)
    exe = str(ROOT / "zig-out" / "bin" / "zsr")
    ok = True
    with tempfile.TemporaryDirectory(dir=CACHE if CACHE.exists() else None) as tmp:
        tmp = Path(tmp)
        for ident, seed in CASES:
            eq = next(e for e in EQUATIONS if e[0] == ident)
            names, x, y = dataset(eq, 1, 1000)
            data = tmp / f"{ident}.csv"
            write_csv(data, names, x, y, eq[1])
            fronts = []
            for k in range(2):
                out = tmp / f"{ident}-{k}.json"
                subprocess.run([exe, str(data), "--generations", str(GENERATIONS), "--seed", str(seed), "--ops", OPS,
                                "--json", str(out)], check=True, capture_output=True)
                fronts.append(json.loads(out.read_text()))
            res = fronts[0]
            best = res["front"][res["selected"]]
            truth = sympy.sympify(eq[2], locals={n: s for n, s in zip(names, sympy.symbols(names, positive=True))})
            found = to_sympy(best["expr"], names)
            exact = bool(is_exact(found, truth))
            losses = [f["nmse"] for f in res["front"]]
            falling = all(b < a for a, b in zip(losses, losses[1:]))
            same = fronts[0]["front"] == fronts[1]["front"]
            good = exact and falling and same
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {ident:8} exact={exact} front_falls={falling} deterministic={same} "
                  f"size={best['size']} generations={res['generations']}  {found}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
