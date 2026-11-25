"""Figures for the README.

    python3 scripts/bench.py      # docs/data/bench.json
    python3 scripts/figures.py
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from common import FIGURES, RESULTS, round_constants, to_sympy  # noqa: E402
from equations import DEVELOPMENT, EQUATIONS, HELD_OUT  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, ORANGE_LIGHT, AQUA_LIGHT = "#d6e6f8", "#fbe0d4", "#d2f0e4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
TOOLS = [("zsr", BLUE), ("gplearn", ORANGE)]


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def load():
    return json.loads((RESULTS / "bench.json").read_text())


def short(text, names):
    """An expression for display: simple constants recognised, others to three digits."""
    e = round_constants(to_sympy(text, names))
    e = e.xreplace({f: sympy.Float(f, 3) for f in e.atoms(sympy.Float)})
    return str(e).replace("**", "^").replace("*", "·")


def front(bench, ident, seed, callouts):
    """The hero: one run's Pareto front, from a single constant to the law."""
    eq = next(e for e in EQUATIONS if e[0] == ident)
    names = list(eq[3])
    run = next(r for r in bench["runs"] if r["tool"] == "zsr" and r["equation"] == ident and r["seed"] == seed)
    fr = []
    for f in run["front"]:  # up to the first exact fit; larger exact ones differ only by rounding
        fr.append(f)
        if f["nmse"] <= 1e-20:
            break
    floor = 1e-32
    sizes = np.array([f["size"] for f in fr])
    losses = np.array([max(f["nmse"], floor) for f in fr])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step(sizes, losses, where="post", color=AXIS, lw=1.2, zorder=1)
    ax.scatter(sizes, losses, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter([sizes[-1]], [losses[-1]], s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("size (tokens)")
    ax.set_ylabel("NMSE on the training data")
    ax.set_ylim(floor / 30, 30)
    ax.set_xlim(0, sizes[-1] + 3)

    def label(i):
        e = round_constants(to_sympy(fr[i]["expr"], names))
        if i == len(fr) - 1:
            e = sympy.factor(sympy.simplify(e))
        txt = str(e.xreplace({f: sympy.Float(f, 3) for f in e.atoms(sympy.Float)}))
        for a, b in (("**", "^"), ("*", "·"), ("omega", "ω"), ("alpha", "α")):
            txt = txt.replace(a, b)
        return txt

    # Callouts in the empty space below the front, with leader lines.
    picks = [i for i, f in enumerate(fr) if f["size"] in callouts] + [len(fr) - 1]
    for k, i in enumerate(picks):
        last = i == len(fr) - 1
        at = (sizes[i] - 0.3, 1e-29) if last else (sizes[i] + 0.4, 10.0 ** (-5 - 6 * k))
        text = label(i) if last else f"{label(i)}    (NMSE {fr[i]['nmse']:.2f})"
        ax.annotate(text, (sizes[i], losses[i]), xytext=at, textcoords="data", ha="right" if last else "left",
                    va="center", fontsize=9 if last else 8.5, color=INK if last else INK2,
                    fontweight="bold" if last else "normal",
                    arrowprops=None if last else dict(arrowstyle="-", color=MUTED, lw=0.7, ls=(0, (3, 2)), shrinkA=2,
                                                      shrinkB=4))
    ax.set_title(f"zsr on {ident} (held out from tuning): x = x1·(cos(ω·t) + α·cos²(ω·t)), "
                 f"the most accurate formula of each size", fontsize=10.5)
    ax.text(0, -0.2, f"Seed 1, 1,000 points. The exact law (orange, NMSE below 1e-32, drawn at 1e-32) was found after "
            f"{run['exact_fit_seconds']:.1f} s on one thread; the front is cut there.", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    save(fig, "front.png")


def solved(bench):
    """Anytime curves: runs solved exactly by each point in time."""
    budget = bench["time_budget"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (title, eqs) in zip(axes, [("Development set", DEVELOPMENT), ("Held-out set", HELD_OUT)]):
        idents = {e[0] for e in eqs}
        total = len(idents) * len(bench["seeds"])
        t = np.logspace(-2, np.log10(budget), 400)
        for tool, col in TOOLS:
            runs = [r for r in bench["runs"] if r["tool"] == tool and r["equation"] in idents]
            times = np.array([r["exact_fit_seconds"] for r in runs if r["exact"]], dtype=float)
            count = np.array([(times <= x).sum() for x in t])
            ax.step(t, count, where="post", color=col, lw=2, label=tool)
            ax.annotate(f"{tool} {count[-1]}/{total}", (t[-1], count[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.01, budget)
        ax.set_ylim(0, total)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("seconds (one thread)")
        ax.set_ylabel(f"runs solved exactly (of {total})")
        ax.set_title(f"{title}: {len(idents)} equations × {len(bench['seeds'])} seeds")
    axes[0].legend(loc="upper left")
    fig.text(0.01, -0.04, "A run counts once its model is exact: equal to the true law after rounding constants, with test "
             "NMSE ≤ 1e-12. Time is when the tool first reached its own exact-fit threshold.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "solved.png")


def speed(bench):
    """Expression-point evaluations per second during the benchmark runs."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    rng = np.random.default_rng(0)
    for k, (tool, col) in enumerate(TOOLS):
        v = np.array([r["point_evaluations"] / r["seconds"] for r in bench["runs"] if r["tool"] == tool and r["seconds"] > 1])
        y = len(TOOLS) - 1 - k
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=col, alpha=0.7, linewidths=0)
        med = float(np.median(v))
        ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, lw=2)
        ax.annotate(f"median {med / 1e6:,.0f} million / s", (med, y + 0.32), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(TOOLS)))
    ax.set_yticklabels([t for t, _ in reversed(TOOLS)])
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(TOOLS) - 0.2)
    ax.set_xlabel("expression evaluations × data points, per second (one thread; runs longer than 1 s)")
    ax.set_title("Evaluation throughput in the benchmark runs")
    ax.grid(axis="y", visible=False)
    save(fig, "speed.png")


def rpn():
    """Diagram: the token array, and the stack of columns while it runs."""
    toks = [("x0", "leaf"), ("c0", "leaf"), ("add", "op"), ("x1", "leaf"), ("sin", "op"), ("mul", "op")]
    fill = {"leaf": BLUE_LIGHT, "op": ORANGE_LIGHT}
    edge = {"leaf": BLUE, "op": ORANGE}
    stacks = [["x0"], ["x0", "1.5"], ["x0+1.5"], ["x0+1.5", "x1"], ["x0+1.5", "sin x1"], ["(x0+1.5)·sin x1"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.45, 7.2)
    ax.axis("off")
    ax.text(0.1, 6.9, "(x0 + 1.5) · sin(x1) is stored as six tokens and one constant, c0 = 1.5", fontsize=11,
            fontweight="bold", color=INK, va="top")
    w, gap, x0 = 1.5, 0.35, 0.6
    for i, (t, kind) in enumerate(toks):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 5.4), w, 0.75, boxstyle="round,pad=0,rounding_size=0.08", fc=fill[kind],
                                    ec=edge[kind], lw=1.2))
        ax.text(x + w / 2, 5.775, t, ha="center", va="center", fontsize=11, family="monospace", color=INK)
    for (a, b, label) in [(0, 2, "subtree: x0 + 1.5"), (3, 4, "subtree: sin(x1)")]:
        xa, xb = x0 + a * (w + gap), x0 + b * (w + gap) + w
        ax.plot([xa, xa, xb, xb], [5.3, 5.15, 5.15, 5.3], color=MUTED, lw=1)
        ax.text((xa + xb) / 2, 5.0, label, ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(0.1, 4.35, "Running it: each token works through whole columns of data before the next token runs",
            fontsize=10, color=INK, va="top")
    ax.text(0.1, 3.95, "(a column holds the value at every data point; each stripe stands for 8 points, one @Vector(8, f64))",
            fontsize=8.5, color=INK2, va="top")
    lane_h = 0.26
    for i, st in enumerate(stacks):
        xc = x0 + i * (w + gap)
        for j, name in enumerate(st):
            colw = 0.62 if len(st) == 2 else 1.2
            xs = xc + (0.08 + j * 0.74 if len(st) == 2 else 0.15)
            for k in range(8):
                ax.add_patch(Rectangle((xs, 1.0 + k * lane_h), colw, lane_h * 0.86, fc=BLUE_LIGHT if j == 0 else AQUA_LIGHT,
                                       ec="none"))
            ax.text(xs + colw / 2, 0.82, name, ha="center", va="top", fontsize=7.5 if len(name) > 7 else 8.5,
                    color=INK2, family="monospace")
        ax.text(xc + w / 2, 3.2, f"after {toks[i][0]}", ha="center", fontsize=8.5, color=INK)
    ax.text(0.1, -0.05, "Every subtree is a contiguous run of tokens, so crossover and mutation are array splices, and a "
            "program is a fixed-size value with no heap allocation.", fontsize=8.5, color=MUTED)
    save(fig, "rpn.png")


def main():
    which = sys.argv[1:] or ["rpn", "front", "solved", "speed"]
    bench = load() if set(which) - {"rpn"} else None
    if "rpn" in which:
        rpn()
    if "front" in which:
        front(bench, "I.50.26", 1, callouts=(1, 8, 13))
    if "solved" in which:
        solved(bench)
    if "speed" in which:
        speed(bench)


if __name__ == "__main__":
    main()
