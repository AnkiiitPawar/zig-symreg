//! Evaluating an expression on a whole data set at once.
//!
//! Each token is applied to every data point before the next token runs: the
//! stack holds whole columns, and each operator is a loop over 8-wide f64
//! vectors. The interpreter's dispatch cost is paid once per token rather
//! than once per token per point.

const std = @import("std");
const expr = @import("expr.zig");
const Allocator = std.mem.Allocator;

pub const lanes = 8;
pub const V = @Vector(lanes, f64);

/// Columns of the variables and the target, each padded to a multiple of 8
/// points by repeating the last point.
pub const Dataset = struct {
    n: usize, // real points
    padded: usize,
    nvars: usize,
    cols: [][]f64, // nvars columns
    y: []f64,
    y_mean: f64,
    y_var: f64,

    pub fn init(gpa: Allocator, xs: []const []const f64, y: []const f64) !Dataset {
        const n = y.len;
        const padded = (n + lanes - 1) / lanes * lanes;
        const cols = try gpa.alloc([]f64, xs.len);
        for (cols, xs) |*c, src| {
            c.* = try gpa.alloc(f64, padded);
            @memcpy(c.*[0..n], src);
            @memset(c.*[n..], src[n - 1]);
        }
        const yy = try gpa.alloc(f64, padded);
        @memcpy(yy[0..n], y);
        @memset(yy[n..], y[n - 1]);
        var mean: f64 = 0;
        for (y) |v| mean += v;
        mean /= @floatFromInt(n);
        var v2: f64 = 0;
        for (y) |v| v2 += (v - mean) * (v - mean);
        v2 /= @floatFromInt(n);
        return .{ .n = n, .padded = padded, .nvars = xs.len, .cols = cols, .y = yy, .y_mean = mean, .y_var = v2 };
    }

    pub fn deinit(d: Dataset, gpa: Allocator) void {
        for (d.cols) |c| gpa.free(c);
        gpa.free(d.cols);
        gpa.free(d.y);
    }

    /// The first `m` points (rounded up to whole vectors), sharing memory.
    pub fn prefix(d: Dataset, m: usize) Dataset {
        if (m >= d.n) return d;
        var s = d;
        s.n = m;
        s.padded = (m + lanes - 1) / lanes * lanes;
        return s;
    }
};

/// Scratch space for one evaluator: a stack of columns.
pub const Scratch = struct {
    stack: [][]align(@alignOf(V)) f64,

    pub fn init(gpa: Allocator, padded: usize) !Scratch {
        const st = try gpa.alloc([]align(@alignOf(V)) f64, expr.max_len);
        for (st) |*c| c.* = try gpa.alignedAlloc(f64, .of(V), padded);
        return .{ .stack = st };
    }

    pub fn deinit(s: Scratch, gpa: Allocator) void {
        for (s.stack) |c| gpa.free(c);
        gpa.free(s.stack);
    }
};

inline fn vec(col: []const f64, i: usize) V {
    return col[i..][0..lanes].*;
}

inline fn put(col: []f64, i: usize, v: V) void {
    col[i..][0..lanes].* = v;
}

/// Evaluates `e` on the first `d.padded` points; returns the output column
/// (inside `s`).
pub fn evaluate(e: *const expr.Expr, d: Dataset, s: Scratch) []f64 {
    var sp: usize = 0;
    const m = d.padded;
    for (e.toks[0..e.len]) |t| {
        switch (t.op) {
            .x => {
                @memcpy(s.stack[sp][0..m], d.cols[t.arg][0..m]);
                sp += 1;
            },
            .c => {
                @memset(s.stack[sp][0..m], e.consts[t.arg]);
                sp += 1;
            },
            .add, .sub, .mul, .div => {
                sp -= 1;
                const a = s.stack[sp - 1];
                const b = s.stack[sp];
                var i: usize = 0;
                switch (t.op) {
                    .add => while (i < m) : (i += lanes) put(a, i, vec(a, i) + vec(b, i)),
                    .sub => while (i < m) : (i += lanes) put(a, i, vec(a, i) - vec(b, i)),
                    .mul => while (i < m) : (i += lanes) put(a, i, vec(a, i) * vec(b, i)),
                    else => while (i < m) : (i += lanes) put(a, i, vec(a, i) / vec(b, i)),
                }
            },
            else => {
                const a = s.stack[sp - 1];
                var i: usize = 0;
                switch (t.op) {
                    .sin => while (i < m) : (i += lanes) put(a, i, @sin(vec(a, i))),
                    .cos => while (i < m) : (i += lanes) put(a, i, @cos(vec(a, i))),
                    .exp => while (i < m) : (i += lanes) put(a, i, @exp(vec(a, i))),
                    .log => while (i < m) : (i += lanes) put(a, i, @log(vec(a, i))),
                    .sqrt => while (i < m) : (i += lanes) put(a, i, @sqrt(vec(a, i))),
                    .square => while (i < m) : (i += lanes) put(a, i, vec(a, i) * vec(a, i)),
                    .neg => while (i < m) : (i += lanes) put(a, i, -vec(a, i)),
                    else => unreachable,
                }
            },
        }
    }
    return s.stack[0][0..m];
}

/// Mean squared error over the real points divided by the variance of y,
/// or +inf if any prediction is not finite.
pub fn nmse(e: *const expr.Expr, d: Dataset, s: Scratch) f64 {
    const out = evaluate(e, d, s);
    var acc: V = @splat(0);
    var i: usize = 0;
    const full = d.n / lanes * lanes;
    while (i < full) : (i += lanes) {
        const r = vec(out, i) - vec(d.y, i);
        acc += r * r;
    }
    var total = @reduce(.Add, acc);
    while (i < d.n) : (i += 1) {
        const r = out[i] - d.y[i];
        total += r * r;
    }
    if (!std.math.isFinite(total)) return std.math.inf(f64);
    const v = total / @as(f64, @floatFromInt(d.n));
    return if (d.y_var > 0) v / d.y_var else v;
}

/// A linear rescaling a + b * e of an expression's output, and the NMSE of
/// the rescaled model.
pub const Scaled = struct {
    loss: f64,
    a: f64 = 0,
    b: f64 = 1,
};

/// Linear scaling (Keijzer 2003): the least-squares a and b for the model
/// a + b * e, in closed form, and the NMSE of that model. Coefficients
/// within 1e-12 of 0, 1 or -1 are set to those values, so an expression that
/// needs no rescaling gets a = 0 and b = 1 and the loss nmse(e). An
/// expression whose output is constant gets b = 0 and a = mean(y).
pub fn scaled(e: *const expr.Expr, d: Dataset, s: Scratch) Scaled {
    const inf = std.math.inf(f64);
    const out = evaluate(e, d, s);
    const n = d.n;
    const nf: f64 = @floatFromInt(n);
    const full = n / lanes * lanes;

    var acc: V = @splat(0);
    var i: usize = 0;
    while (i < full) : (i += lanes) acc += vec(out, i);
    var sum = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += out[i];
    if (!std.math.isFinite(sum)) return .{ .loss = inf };
    const fm = sum / nf;

    const fmv: V = @splat(fm);
    const ymv: V = @splat(d.y_mean);
    var aff: V = @splat(0);
    var afy: V = @splat(0);
    i = 0;
    while (i < full) : (i += lanes) {
        const df = vec(out, i) - fmv;
        aff += df * df;
        afy += df * (vec(d.y, i) - ymv);
    }
    var sff = @reduce(.Add, aff);
    var sfy = @reduce(.Add, afy);
    while (i < n) : (i += 1) {
        const df = out[i] - fm;
        sff += df * df;
        sfy += df * (d.y[i] - d.y_mean);
    }
    if (!std.math.isFinite(sff) or !std.math.isFinite(sfy)) return .{ .loss = inf };
    const syy = nf * d.y_var;
    const norm = if (d.y_var > 0) syy else nf;
    if (!(sff > 1e-24 * fm * fm * nf)) return .{ .loss = syy / norm, .a = d.y_mean, .b = 0 };

    var b = sfy / sff;
    var a = d.y_mean - b * fm;
    const tol = 1e-12;
    if (@abs(b - 1) <= tol) b = 1 else if (@abs(b + 1) <= tol) b = -1;
    if (@abs(a) <= tol * (@abs(d.y_mean) + @sqrt(d.y_var))) a = 0;

    // The sum of squared residuals follows from the sums above for any a
    // and b, but cancels near an exact fit: there, add up the residuals.
    const off = a + b * fm - d.y_mean;
    var sse = nf * off * off + b * b * sff - 2 * b * sfy + syy;
    if (!(sse > 1e-6 * syy)) {
        const av: V = @splat(a);
        const bv: V = @splat(b);
        var ar: V = @splat(0);
        i = 0;
        while (i < full) : (i += lanes) {
            const r = av + bv * vec(out, i) - vec(d.y, i);
            ar += r * r;
        }
        sse = @reduce(.Add, ar);
        while (i < n) : (i += 1) {
            const r = a + b * out[i] - d.y[i];
            sse += r * r;
        }
    }
    return .{ .loss = @max(sse, 0) / norm, .a = a, .b = b };
}

const testing = std.testing;

test "column evaluation matches a scalar loop" {
    const gpa = testing.allocator;
    var x0: [37]f64 = undefined;
    var x1: [37]f64 = undefined;
    var y: [37]f64 = undefined;
    for (0..37) |i| {
        const t: f64 = @floatFromInt(i);
        x0[i] = 1 + t * 0.1;
        x1[i] = 2 - t * 0.05;
        y[i] = (x0[i] + 1.5) * @sin(x1[i]);
    }
    const d = try Dataset.init(gpa, &.{ &x0, &x1 }, &y);
    defer d.deinit(gpa);
    const s = try Scratch.init(gpa, d.padded);
    defer s.deinit(gpa);
    var e: expr.Expr = .{ .len = 6, .nconst = 1 };
    e.toks[0..6].* = .{ .{ .op = .x, .arg = 0 }, .{ .op = .c, .arg = 0 }, .{ .op = .add }, .{ .op = .x, .arg = 1 }, .{ .op = .sin }, .{ .op = .mul } };
    e.consts[0] = 1.5;
    const out = evaluate(&e, d, s);
    for (0..37) |i| try testing.expectApproxEqAbs(y[i], out[i], 1e-14);
    try testing.expectEqual(@as(f64, 0), nmse(&e, d, s));
    e.consts[0] = 1.0;
    try testing.expect(nmse(&e, d, s) > 0);
}

test "linear scaling finds the best a and b" {
    const gpa = testing.allocator;
    var x0: [53]f64 = undefined;
    var y: [53]f64 = undefined;
    for (0..53) |i| {
        x0[i] = 1 + @as(f64, @floatFromInt(i)) * 0.07;
        y[i] = 3 - 0.25 * @sin(x0[i]);
    }
    const d = try Dataset.init(gpa, &.{&x0}, &y);
    defer d.deinit(gpa);
    const s = try Scratch.init(gpa, d.padded);
    defer s.deinit(gpa);
    var e: expr.Expr = .{ .len = 2 };
    e.toks[0..2].* = .{ .{ .op = .x, .arg = 0 }, .{ .op = .sin } };
    const r = scaled(&e, d, s);
    try testing.expectApproxEqAbs(@as(f64, 3), r.a, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.25), r.b, 1e-12);
    try testing.expect(r.loss < 1e-28);
    // An expression that needs no rescaling reports none.
    for (&y, x0) |*v, x| v.* = @sin(x);
    const d2 = try Dataset.init(gpa, &.{&x0}, &y);
    defer d2.deinit(gpa);
    const r2 = scaled(&e, d2, s);
    try testing.expectEqual(@as(f64, 0), r2.a);
    try testing.expectEqual(@as(f64, 1), r2.b);
    try testing.expectEqual(nmse(&e, d2, s), r2.loss);
    // A constant output: the model is the mean of y.
    e.toks[0] = .{ .op = .c };
    e.nconst = 1;
    e.consts[0] = 2;
    e.len = 1;
    const r3 = scaled(&e, d2, s);
    try testing.expectEqual(@as(f64, 0), r3.b);
    try testing.expectApproxEqAbs(@as(f64, 1), r3.loss, 1e-12);
}
