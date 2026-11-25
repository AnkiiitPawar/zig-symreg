"""Shared helpers: data generation, running zsr and gplearn, and checking
whether a found expression is the true law."""
import json
import zlib
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import sympy

from equations import EQUATIONS

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("ZSR_CACHE") or ROOT / ".cache")
RESULTS = ROOT / "docs" / "data"
FIGURES = ROOT / "docs"
N_TRAIN, N_TEST = 1000, 1000


def zsr_binary():
    exe = ROOT / "zig-out" / "bin" / "zsr"
    if not exe.exists():
        subprocess.run(shlex.split(os.environ.get("ZIG", "zig")) + ["build", "-Doptimize=ReleaseFast"], cwd=ROOT, check=True)
    return exe


def dataset(eq, seed, n):
    """Points drawn uniformly from the equation's ranges, and the exact target."""
    ident, target, formula, ranges = eq
    rng = np.random.default_rng([seed, zlib.crc32(ident.encode())])
    names = list(ranges)
    syms = sympy.symbols(names)
    f = sympy.lambdify(syms, sympy.sympify(formula, locals=dict(zip(names, syms))), "numpy")
    cols = [rng.uniform(lo, hi, n) for lo, hi in ranges.values()]
    return names, np.column_stack(cols), np.asarray(f(*cols), dtype=np.float64)


def write_csv(path, names, x, y, target):
    with open(path, "w") as fh:
        fh.write(",".join(names + [target]) + "\n")
        for row, t in zip(x, y):
            fh.write(",".join(repr(float(v)) for v in row) + "," + repr(float(t)) + "\n")


def symbols(names):
    """SymPy symbols for the variables; all are positive in every equation's
    range, which lets SymPy simplify sqrt(x**2) to x."""
    return sympy.symbols(names, positive=True)


def to_sympy(text, names):
    """Parse an expression in x0, x1, ... into SymPy with the real variable names."""
    syms = symbols(names)
    local = {f"x{i}": s for i, s in enumerate(syms)}
    local.update({"sqrt": sympy.sqrt, "exp": sympy.exp, "log": sympy.log, "sin": sympy.sin, "cos": sympy.cos})
    return sympy.sympify(text, locals=local)


def round_constants(e):
    """Replace each float by a nearby simple number, when one is within 1e-6
    relative: a ratio of integers up to 64, possibly times pi, sqrt(pi) or
    sqrt(2 pi) or divided by one of them. Floats below 1e-10 in size become 0."""
    pi = sympy.pi
    bases = (1, pi, 1 / pi, sympy.sqrt(pi), 1 / sympy.sqrt(pi), sympy.sqrt(2 * pi), 1 / sympy.sqrt(2 * pi))
    reps = {}
    for c in e.atoms(sympy.Float):
        v = float(c)
        if abs(v) < 1e-10:
            reps[c] = 0
            continue
        best = None
        for base in bases:
            r = sympy.nsimplify(v / float(base), rational=True, tolerance=1e-9)
            if abs(float(r * base) - v) <= 1e-6 * abs(v) and sympy.denom(r) <= 64 and abs(sympy.numer(r)) <= 64:
                best = r * base
                break
        if best is not None:
            reps[c] = best
    return e.xreplace(reps)


def is_exact(found, truth):
    """True if `found` equals `truth` once constants are rounded: their
    difference simplifies to 0, or their ratio to 1. Rounding is applied to
    the difference and ratio too, so that two copies of a constant that
    differ in the last bit, which cancel in exact arithmetic, leave no
    residue."""
    try:
        f = round_constants(sympy.simplify(round_constants(found)))
        if round_constants(sympy.simplify(f - truth)) == 0:
            return True
        return round_constants(sympy.simplify(f / truth)) == 1
    except Exception:
        return False


def r2(pred, y):
    return 1 - np.mean((pred - y) ** 2) / np.var(y)


def save_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n")
