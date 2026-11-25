//! Genetic programming for symbolic regression.
//!
//! Several islands evolve independently. Every generation, each island makes
//! a new population by tournament selection on a score that trades accuracy
//! against size, and applies crossover, subtree mutation, point mutation,
//! constant mutation and hoisting. Each expression e is judged as a + b * e
//! with the best a and b (linear scaling), so the search looks for the shape
//! of the law and leaves its scale to least squares. Constants of some
//! children, and of every new hall-of-fame entry, are fitted by
//! Levenberg-Marquardt. A hall of fame keeps the most accurate model of each
//! size, with the scaling written into the expression and counted in its
//! size; its Pareto front is the result, and it also seeds migration back
//! into the islands.

const std = @import("std");
const expr = @import("expr.zig");
const eval = @import("eval.zig");
const fit = @import("fit.zig");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    islands: usize = 4,
    population: usize = 250,
    threads: usize = 1,
    tournament: usize = 6,
    max_depth_init: usize = 4,
    /// Added to ln(loss) per token when comparing individuals. Island i uses
    /// parsimony[i % parsimony.len], so islands can differ.
    parsimony: []const f64 = &.{0.03},
    p_crossover: f64 = 0.4,
    p_subtree: f64 = 0.2,
    p_point: f64 = 0.15,
    p_const: f64 = 0.1,
    p_hoist: f64 = 0.05,
    /// Insertion: a random subtree s becomes op(s, r), op(r, s) or op(s) for
    /// a new random tree r.
    p_insert: f64 = 0.1,
    /// Probability that a child's constants are fitted.
    optimize_probability: f64 = 0.1,
    /// Constants are fitted on this many points, for at most this many
    /// Levenberg-Marquardt iterations.
    fit_points: usize = 128,
    fit_iterations: usize = 15,
    migrate_every: usize = 50,
    /// Judge each expression e as a + b * e with the best a and b.
    linear_scaling: bool = true,
    max_generations: usize = 1_000_000,
    time_limit: f64 = 30,
    /// A loss (NMSE) at or below this counts as an exact fit.
    exact_nmse: f64 = 1e-20,
    /// Generations to keep going once an exact fit is found, to look for a
    /// smaller one.
    after_exact: usize = 30,
    seed: u64 = 1,
    ops: expr.OpSet = .{},
};

pub const Individual = struct {
    e: expr.Expr,
    loss: f64,
    /// The model is a + b * e. Hall-of-fame entries have the scaling written
    /// into e, with a = 0 and b = 1.
    a: f64 = 0,
    b: f64 = 1,
    fitted: bool = false,

    fn score(ind: *const Individual, parsimony: f64) f64 {
        if (!std.math.isFinite(ind.loss)) return std.math.inf(f64);
        return @log(@max(ind.loss, 1e-30)) + parsimony * @as(f64, @floatFromInt(ind.e.len));
    }
};

/// The most accurate expression found for each size.
pub const HallOfFame = struct {
    best: [expr.max_len + 1]?Individual = @splat(null),

    fn improves(h: *const HallOfFame, size: usize, loss: f64) bool {
        if (h.best[size]) |cur| return loss < cur.loss;
        return std.math.isFinite(loss);
    }

    fn offer(h: *HallOfFame, ind: Individual) bool {
        if (!h.improves(ind.e.len, ind.loss)) return false;
        h.best[ind.e.len] = ind;
        return true;
    }

    pub fn bestLoss(h: *const HallOfFame) f64 {
        var b = std.math.inf(f64);
        for (h.best) |s| if (s) |ind| {
            b = @min(b, ind.loss);
        };
        return b;
    }

    /// Entries more accurate than every smaller one, by increasing size.
    pub fn front(h: *const HallOfFame, out: []Individual) []Individual {
        var n: usize = 0;
        var best = std.math.inf(f64);
        for (h.best) |s| if (s) |ind| {
            if (ind.loss < best) {
                out[n] = ind;
                n += 1;
                best = ind.loss;
            }
        };
        return out[0..n];
    }
};

/// The model chosen from the front: the one with the largest drop in log loss
/// per extra token (PySR's score) among those within 1.5x of the best loss,
/// with losses below the exact-fit threshold treated as equal.
pub fn select(front: []const Individual, exact_nmse: f64) usize {
    var min_l = std.math.inf(f64);
    for (front) |f| min_l = @min(min_l, @max(f.loss, exact_nmse));
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (front, 0..) |f, i| {
        const l = @max(f.loss, exact_nmse);
        if (l > 1.5 * min_l) continue;
        const s: f64 = if (i == 0) std.math.inf(f64) else blk: {
            const prev = @max(front[i - 1].loss, exact_nmse);
            break :blk -(@log(l) - @log(prev)) / @as(f64, @floatFromInt(f.e.len - front[i - 1].e.len));
        };
        if (s > best_score) {
            best_score = s;
            best = i;
        }
    }
    return best;
}

const Island = struct {
    pop: []Individual,
    next: []Individual,
    prng: std.Random.DefaultPrng,
    parsimony: f64,
    hof: HallOfFame = .{},
    scratch: eval.Scratch,
    ws: fit.Workspace,
    evals: u64 = 0,
    lm_calls: u64 = 0,
    lm_evals: u64 = 0,
    failed: bool = false,
};

pub const Result = struct {
    front: []Individual,
    selected: usize,
    generations: usize,
    evaluations: u64,
    lm_calls: u64,
    lm_evaluations: u64,
    seconds: f64,
    exact_at: ?f64, // seconds until the first exact fit, if any
};

const Ctx = struct {
    d: eval.Dataset,
    cfg: Config,
};

fn tournament(isl: *Island, cfg: Config) *const Individual {
    const r = isl.prng.random();
    var best = &isl.pop[r.uintLessThan(usize, isl.pop.len)];
    for (1..cfg.tournament) |_| {
        const c = &isl.pop[r.uintLessThan(usize, isl.pop.len)];
        if (c.score(isl.parsimony) < best.score(isl.parsimony)) best = c;
    }
    return best;
}

fn randomSubtree(e: *const expr.Expr, r: std.Random) [2]usize {
    const end = r.uintLessThan(usize, e.len);
    return .{ e.subtreeStart(end), end };
}

fn child(isl: *Island, ctx: *const Ctx) expr.Expr {
    const cfg = ctx.cfg;
    const r = isl.prng.random();
    const nv = ctx.d.nvars;
    const u = r.float(f64);
    var t = cfg.p_crossover;
    if (u < t) {
        const a = tournament(isl, cfg);
        const b = tournament(isl, cfg);
        const sa = randomSubtree(&a.e, r);
        const sb = randomSubtree(&b.e, r);
        return expr.splice(&a.e, sa[0], sa[1], &b.e, sb[0], sb[1]) orelse a.e;
    }
    t += cfg.p_subtree;
    if (u < t) {
        const a = tournament(isl, cfg);
        const sa = randomSubtree(&a.e, r);
        const fresh = expr.random(r, nv, cfg.ops, 3);
        return expr.splice(&a.e, sa[0], sa[1], &fresh, 0, fresh.len - 1) orelse a.e;
    }
    t += cfg.p_point;
    if (u < t) {
        var e = tournament(isl, cfg).e;
        const i = r.uintLessThan(usize, e.len);
        const tok = &e.toks[i];
        switch (tok.op.arity()) {
            0 => {
                if (r.float(f64) < 0.25 and e.nconst < expr.max_consts) {
                    e.consts[e.nconst] = r.floatNorm(f64) * 2;
                    tok.* = .{ .op = .c, .arg = e.nconst };
                    e.nconst += 1;
                } else tok.* = .{ .op = .x, .arg = @intCast(r.uintLessThan(usize, nv)) };
                e.compactConsts();
            },
            1 => tok.op = cfg.ops.unary[r.uintLessThan(usize, cfg.ops.unary.len)],
            else => tok.op = cfg.ops.binary[r.uintLessThan(usize, cfg.ops.binary.len)],
        }
        return e;
    }
    t += cfg.p_const;
    if (u < t) {
        var e = tournament(isl, cfg).e;
        if (e.nconst == 0) return e;
        const j = r.uintLessThan(usize, e.nconst);
        if (r.boolean()) e.consts[j] *= 1 + 0.2 * r.floatNorm(f64) else e.consts[j] += r.floatNorm(f64);
        return e;
    }
    t += cfg.p_hoist;
    if (u < t) {
        const a = tournament(isl, cfg);
        const sa = randomSubtree(&a.e, r);
        return expr.splice(&a.e, 0, a.e.len - 1, &a.e, sa[0], sa[1]) orelse a.e;
    }
    t += cfg.p_insert;
    if (u < t) {
        const a = tournament(isl, cfg);
        const sa = randomSubtree(&a.e, r);
        const fresh = expr.random(r, nv, cfg.ops, 2);
        const nb = cfg.ops.binary.len;
        const k = r.uintLessThan(usize, nb + cfg.ops.unary.len);
        const op = if (k < nb) cfg.ops.binary[k] else cfg.ops.unary[k - nb];
        return expr.insert(&a.e, sa[0], sa[1], &fresh, op, r.boolean()) orelse a.e;
    }
    return expr.random(r, nv, cfg.ops, cfg.max_depth_init);
}

/// Fits the constants of a + b * e, with a and b, on the first `fit_points`
/// data points (enough for data without noise, and much cheaper), then
/// rescales and scores on all of them. Keeps the result if it is better.
fn fitConstants(isl: *Island, ind: *Individual, d: eval.Dataset, cfg: Config) void {
    if (ind.e.nconst == 0 or ind.fitted) return;
    ind.fitted = true;
    isl.lm_calls += 1;
    const sub = d.prefix(cfg.fit_points);
    const opts: fit.Options = .{ .evals = &isl.lm_evals, .max_iters = cfg.fit_iterations };
    const start: [2]f64 = if (ind.b == 0) .{ 0, 1 } else .{ ind.a, ind.b };
    var best = ind.e;
    var scale = start;
    const best_loss = fit.levenbergMarquardt(&best, sub, isl.scratch, &isl.ws, withScale(opts, &scale, cfg));
    // A second start from perturbed constants, unless the first fit is exact.
    if (!(best_loss <= cfg.exact_nmse)) {
        var e = ind.e;
        const r = isl.prng.random();
        for (e.consts[0..e.nconst]) |*c| c.* = c.* * (1 + 0.5 * r.floatNorm(f64)) + 0.1 * r.floatNorm(f64);
        var sc2 = start;
        const l = fit.levenbergMarquardt(&e, sub, isl.scratch, &isl.ws, withScale(opts, &sc2, cfg));
        if (l < best_loss) best = e;
    }
    const sc = judge(&best, d, isl.scratch, cfg);
    isl.evals += 1;
    if (sc.loss <= ind.loss) ind.* = .{ .e = best, .loss = sc.loss, .a = sc.a, .b = sc.b, .fitted = true };
}

fn withScale(opts: fit.Options, scale: *[2]f64, cfg: Config) fit.Options {
    var o = opts;
    if (cfg.linear_scaling) o.scale = scale;
    return o;
}

/// The loss of an expression, with its linear scaling if enabled.
fn judge(e: *const expr.Expr, d: eval.Dataset, s: eval.Scratch, cfg: Config) eval.Scaled {
    return if (cfg.linear_scaling) eval.scaled(e, d, s) else .{ .loss = eval.nmse(e, d, s) };
}

/// Offers an individual to the island's hall of fame as the expression
/// a + b * e. A new entry has its constants fitted first.
fn offer(isl: *Island, ind: *Individual, d: eval.Dataset, cfg: Config) void {
    if (!std.math.isFinite(ind.loss)) return;
    var m = expr.materialise(&ind.e, ind.a, ind.b) orelse return;
    if (!isl.hof.improves(m.len, ind.loss)) return;
    if (!ind.fitted and ind.e.nconst > 0) {
        fitConstants(isl, ind, d, cfg);
        m = expr.materialise(&ind.e, ind.a, ind.b) orelse return;
    }
    expr.simplify(&m);
    _ = isl.hof.offer(.{ .e = m, .loss = eval.nmse(&m, d, isl.scratch), .fitted = true });
    isl.evals += 1;
}

fn evolve(isl: *Island, ctx: *const Ctx) void {
    const cfg = ctx.cfg;
    const d = ctx.d;
    const r = isl.prng.random();
    // Elitism: keep the best individual.
    var best: usize = 0;
    for (isl.pop, 0..) |*ind, i| {
        if (ind.score(isl.parsimony) < isl.pop[best].score(isl.parsimony)) best = i;
    }
    isl.next[0] = isl.pop[best];
    for (isl.next[1..]) |*slot| {
        var e = child(isl, ctx);
        expr.simplify(&e);
        const sc = judge(&e, d, isl.scratch, cfg);
        var ind: Individual = .{ .e = e, .loss = sc.loss, .a = sc.a, .b = sc.b };
        isl.evals += 1;
        if (ind.e.nconst > 0 and r.float(f64) < cfg.optimize_probability) fitConstants(isl, &ind, d, cfg);
        offer(isl, &ind, d, cfg);
        slot.* = ind;
    }
    std.mem.swap([]Individual, &isl.pop, &isl.next);
}

fn evolveMany(islands: []Island, ctx: *const Ctx, t: usize, stride: usize) void {
    var i = t;
    while (i < islands.len) : (i += stride) evolve(&islands[i], ctx);
}

fn seconds(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e9;
}

pub fn run(gpa: Allocator, io: std.Io, d: eval.Dataset, cfg: Config) !Result {
    const t0 = std.Io.Timestamp.now(io, .awake);
    var master = std.Random.DefaultPrng.init(cfg.seed);
    const islands = try gpa.alloc(Island, cfg.islands);
    defer gpa.free(islands);
    for (islands, 0..) |*isl, i| {
        isl.* = .{
            .parsimony = cfg.parsimony[i % cfg.parsimony.len],
            .pop = try gpa.alloc(Individual, cfg.population),
            .next = try gpa.alloc(Individual, cfg.population),
            .prng = std.Random.DefaultPrng.init(master.random().int(u64)),
            .scratch = try eval.Scratch.init(gpa, d.padded),
            .ws = try fit.Workspace.init(gpa, d.padded),
        };
    }
    defer for (islands) |*isl| {
        gpa.free(isl.pop);
        gpa.free(isl.next);
        isl.scratch.deinit(gpa);
        isl.ws.deinit(gpa);
    };

    const ctx: Ctx = .{ .d = d, .cfg = cfg };
    for (islands) |*isl| {
        const r = isl.prng.random();
        for (isl.pop) |*ind| {
            const e = expr.random(r, d.nvars, cfg.ops, cfg.max_depth_init);
            const sc = judge(&e, d, isl.scratch, cfg);
            ind.* = .{ .e = e, .loss = sc.loss, .a = sc.a, .b = sc.b };
            isl.evals += 1;
            offer(isl, ind, d, cfg);
        }
    }

    var hof: HallOfFame = .{};
    var gen: usize = 0;
    var exact_gen: ?usize = null;
    var exact_at: ?f64 = null;
    while (gen < cfg.max_generations) : (gen += 1) {
        // One generation on every island, spread over the threads.
        const t_count = @max(1, @min(cfg.threads, islands.len));
        var handles: [64]std.Thread = undefined;
        for (0..t_count) |t| {
            if (t + 1 == t_count) evolveMany(islands, &ctx, t, t_count) else handles[t] = try std.Thread.spawn(.{}, evolveMany, .{ islands, &ctx, t, t_count });
        }
        for (handles[0 .. t_count - 1]) |h| h.join();

        // Merge the islands' halls of fame, in island order.
        for (islands) |*isl| for (isl.hof.best) |s| if (s) |ind| {
            _ = hof.offer(ind);
        };
        if (exact_gen == null and hof.bestLoss() <= cfg.exact_nmse) {
            exact_gen = gen;
            exact_at = seconds(io, t0);
        }
        if (exact_gen) |g| if (gen >= g + cfg.after_exact) break;
        if (seconds(io, t0) >= cfg.time_limit) break;

        // Migration: hall-of-fame entries replace random individuals.
        if ((gen + 1) % cfg.migrate_every == 0) {
            var buf: [expr.max_len + 1]Individual = undefined;
            const fr = hof.front(&buf);
            for (islands) |*isl| {
                const r = isl.prng.random();
                for (0..@max(1, cfg.population / 20)) |_| {
                    isl.pop[r.uintLessThan(usize, isl.pop.len)] = fr[r.uintLessThan(usize, fr.len)];
                }
            }
        }
    }

    var buf: [expr.max_len + 1]Individual = undefined;
    const fr = hof.front(&buf);
    const out = try gpa.dupe(Individual, fr);
    var evals: u64 = 0;
    var lm_calls: u64 = 0;
    var lm_evals: u64 = 0;
    for (islands) |isl| {
        evals += isl.evals;
        lm_calls += isl.lm_calls;
        lm_evals += isl.lm_evals;
    }
    return .{ .front = out, .selected = select(out, cfg.exact_nmse), .generations = gen + 1, .evaluations = evals, .lm_calls = lm_calls, .lm_evaluations = lm_evals, .seconds = seconds(io, t0), .exact_at = exact_at };
}

const testing = std.testing;

test "finds a simple law exactly" {
    const gpa = testing.allocator;
    const n = 100;
    var x0: [n]f64 = undefined;
    var x1: [n]f64 = undefined;
    var y: [n]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (0..n) |i| {
        x0[i] = 1 + 4 * prng.random().float(f64);
        x1[i] = 1 + 4 * prng.random().float(f64);
        y[i] = 0.5 * x0[i] * x1[i] * x1[i];
    }
    const d = try eval.Dataset.init(gpa, &.{ &x0, &x1 }, &y);
    defer d.deinit(gpa);
    const res = try run(gpa, testing.io, d, .{ .time_limit = 20, .seed = 3, .islands = 2, .population = 200 });
    defer gpa.free(res.front);
    const best = res.front[res.selected];
    try testing.expect(best.loss <= 1e-20);
    try testing.expect(best.e.len <= 7);
}
