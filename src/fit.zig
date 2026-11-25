//! Fitting an expression's constants by Levenberg-Marquardt, with the
//! Jacobian from forward differences. The model is either the expression
//! itself or a linear rescaling a + b * e of it, whose two coefficients are
//! then fitted with the constants.

const std = @import("std");
const expr = @import("expr.zig");
const eval = @import("eval.zig");
const Allocator = std.mem.Allocator;

/// The most parameters a fit can have: the constants, a and b.
pub const max_params = expr.max_consts + 2;

/// Buffers for one fitter: Jacobian columns, residuals, and the output of
/// the expression at the current constants.
pub const Workspace = struct {
    jac: [max_params][]f64,
    resid: []f64,
    out: []f64,

    pub fn init(gpa: Allocator, padded: usize) !Workspace {
        var w: Workspace = undefined;
        for (&w.jac) |*j| j.* = try gpa.alloc(f64, padded);
        w.resid = try gpa.alloc(f64, padded);
        w.out = try gpa.alloc(f64, padded);
        return w;
    }

    pub fn deinit(w: Workspace, gpa: Allocator) void {
        for (w.jac) |j| gpa.free(j);
        gpa.free(w.resid);
        gpa.free(w.out);
    }
};

pub const Options = struct {
    max_iters: usize = 40,
    /// Incremented by the number of evaluations made, if given.
    evals: ?*u64 = null,
    /// If given, the model is scale[0] + scale[1] * e, and both coefficients
    /// are fitted from the values given.
    scale: ?*[2]f64 = null,
};

/// Improves e.consts (and the scale, if given) in place to reduce the squared
/// error on `d`; returns the final NMSE.
pub fn levenbergMarquardt(e: *expr.Expr, d: eval.Dataset, s: eval.Scratch, w: *Workspace, opts: Options) f64 {
    const nc: usize = e.nconst;
    const scaled = opts.scale != null;
    const k = nc + @as(usize, if (scaled) 2 else 0);
    const n = d.n;
    const norm = @as(f64, @floatFromInt(n)) * (if (d.y_var > 0) d.y_var else 1);
    var a: f64 = 0;
    var b: f64 = 1;
    if (opts.scale) |sc| {
        a = sc[0];
        b = sc[1];
    }
    const first = modelError(e, d, s, a, b);
    if (opts.evals) |c| c.* += 1;
    if (k == 0 or !std.math.isFinite(first.sse)) return first.sse / norm;
    var cost = first.sse;
    keep(w, first.out, d, a, b);

    var lambda: f64 = 1e-3;
    for (0..opts.max_iters) |_| {
        // Jacobian: forward differences for the constants, and the exact
        // columns 1 and e for a and b.
        for (0..nc) |j| {
            const c0 = e.consts[j];
            const h = 1e-7 * @max(1.0, @abs(c0));
            e.consts[j] = c0 + h;
            const o = eval.evaluate(e, d, s);
            if (opts.evals) |c| c.* += 1;
            for (0..n) |i| w.jac[j][i] = b * (o[i] - w.out[i]) / h;
            e.consts[j] = c0;
        }
        if (scaled) {
            @memset(w.jac[nc][0..n], 1);
            @memcpy(w.jac[nc + 1][0..n], w.out[0..n]);
        }
        // Normal equations.
        var jtj: [max_params][max_params]f64 = undefined;
        var jtr: [max_params]f64 = undefined;
        for (0..k) |p| {
            for (p..k) |q| {
                var acc: f64 = 0;
                for (0..n) |i| acc += w.jac[p][i] * w.jac[q][i];
                jtj[p][q] = acc;
                jtj[q][p] = acc;
            }
            var acc: f64 = 0;
            for (0..n) |i| acc += w.jac[p][i] * w.resid[i];
            jtr[p] = acc;
        }
        const saved = e.consts;
        const saved_a = a;
        const saved_b = b;
        var improved = false;
        while (lambda < 1e10) {
            var m: [max_params][max_params]f64 = jtj;
            for (0..k) |p| m[p][p] += lambda * @max(jtj[p][p], 1e-12);
            var step: [max_params]f64 = undefined;
            if (!solve(k, &m, &jtr, &step)) {
                lambda *= 4;
                continue;
            }
            for (0..nc) |p| e.consts[p] = saved[p] - step[p];
            if (scaled) {
                a = saved_a - step[nc];
                b = saved_b - step[nc + 1];
            }
            const trial = modelError(e, d, s, a, b);
            if (opts.evals) |c| c.* += 1;
            if (trial.sse < cost) {
                cost = trial.sse;
                keep(w, trial.out, d, a, b);
                lambda = @max(lambda / 3, 1e-12);
                improved = true;
                break;
            }
            e.consts = saved;
            a = saved_a;
            b = saved_b;
            lambda *= 4;
        }
        if (!improved or cost < 1e-30 * norm) break;
    }
    if (opts.scale) |sc| sc.* = .{ a, b };
    return cost / norm;
}

const ModelError = struct { sse: f64, out: []f64 };

/// Squared error of a + b * e; also returns the output of e (inside `s`).
fn modelError(e: *const expr.Expr, d: eval.Dataset, s: eval.Scratch, a: f64, b: f64) ModelError {
    const out = eval.evaluate(e, d, s);
    var total: f64 = 0;
    for (0..d.n) |i| {
        const r = a + b * out[i] - d.y[i];
        total += r * r;
    }
    return .{ .sse = if (std.math.isFinite(total)) total else std.math.inf(f64), .out = out };
}

/// Records the output of e and the residuals of a + b * e.
fn keep(w: *Workspace, out: []const f64, d: eval.Dataset, a: f64, b: f64) void {
    for (0..d.n) |i| {
        w.out[i] = out[i];
        w.resid[i] = a + b * out[i] - d.y[i];
    }
}

/// Solves the k x k symmetric positive definite system m x = b by Cholesky.
fn solve(k: usize, m: *[max_params][max_params]f64, b: *const [max_params]f64, x: *[max_params]f64) bool {
    var l: [max_params][max_params]f64 = undefined;
    for (0..k) |i| {
        for (0..i + 1) |j| {
            var s = m[i][j];
            for (0..j) |p| s -= l[i][p] * l[j][p];
            if (i == j) {
                if (!(s > 0)) return false;
                l[i][i] = @sqrt(s);
            } else l[i][j] = s / l[j][j];
        }
    }
    var z: [max_params]f64 = undefined;
    for (0..k) |i| {
        var s = b[i];
        for (0..i) |p| s -= l[i][p] * z[p];
        z[i] = s / l[i][i];
    }
    var i: usize = k;
    while (i > 0) {
        i -= 1;
        var s = z[i];
        for (i + 1..k) |p| s -= l[p][i] * x[p];
        x[i] = s / l[i][i];
    }
    return true;
}

const testing = std.testing;

const Fixture = struct {
    d: eval.Dataset,
    s: eval.Scratch,
    w: Workspace,

    fn init(gpa: Allocator, x0: []const f64, y: []const f64) !Fixture {
        const d = try eval.Dataset.init(gpa, &.{x0}, y);
        return .{ .d = d, .s = try eval.Scratch.init(gpa, d.padded), .w = try Workspace.init(gpa, d.padded) };
    }

    fn deinit(f: Fixture, gpa: Allocator) void {
        f.w.deinit(gpa);
        f.s.deinit(gpa);
        f.d.deinit(gpa);
    }
};

test "Levenberg-Marquardt recovers the constants of a nonlinear model" {
    const gpa = testing.allocator;
    const n = 200;
    var x0: [n]f64 = undefined;
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        x0[i] = 1 + 4 * @as(f64, @floatFromInt(i)) / n;
        y[i] = 2.5 * @exp(-0.7 * x0[i]) + 0.3;
    }
    var f = try Fixture.init(gpa, &x0, &y);
    defer f.deinit(gpa);
    // c0 * exp(c1 * x0) + c2, from a poor start.
    var e: expr.Expr = .{ .len = 8, .nconst = 3 };
    e.toks[0..8].* = .{ .{ .op = .c, .arg = 0 }, .{ .op = .c, .arg = 1 }, .{ .op = .x, .arg = 0 }, .{ .op = .mul }, .{ .op = .exp }, .{ .op = .mul }, .{ .op = .c, .arg = 2 }, .{ .op = .add } };
    e.consts[0..3].* = .{ 1, -0.2, 0 };
    const loss = levenbergMarquardt(&e, f.d, f.s, &f.w, .{});
    try testing.expect(loss < 1e-24);
    try testing.expectApproxEqRel(@as(f64, 2.5), e.consts[0], 1e-8);
    try testing.expectApproxEqRel(@as(f64, -0.7), e.consts[1], 1e-8);
    try testing.expectApproxEqRel(@as(f64, 0.3), e.consts[2], 1e-8);

    // The same law as a + b * exp(c0 * x0), fitting a and b too.
    var g: expr.Expr = .{ .len = 4, .nconst = 1 };
    g.toks[0..4].* = .{ .{ .op = .c, .arg = 0 }, .{ .op = .x, .arg = 0 }, .{ .op = .mul }, .{ .op = .exp } };
    g.consts[0] = -0.2;
    var scale: [2]f64 = .{ 0, 1 };
    const loss2 = levenbergMarquardt(&g, f.d, f.s, &f.w, .{ .scale = &scale });
    try testing.expect(loss2 < 1e-24);
    try testing.expectApproxEqRel(@as(f64, -0.7), g.consts[0], 1e-8);
    try testing.expectApproxEqRel(@as(f64, 0.3), scale[0], 1e-8);
    try testing.expectApproxEqRel(@as(f64, 2.5), scale[1], 1e-8);
}
