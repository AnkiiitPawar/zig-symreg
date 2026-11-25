//! Expressions in reverse Polish notation, stored in a fixed-size array.
//!
//! A program such as (x0 + 1.5) * sin(x1) is the token sequence
//!   x0 c0 add x1 sin mul     with constants c0 = 1.5.
//! Every subtree is a contiguous run of tokens, which makes crossover and
//! mutation simple slice operations and lets an individual live in a plain
//! value with no heap allocation.

const std = @import("std");

pub const max_len = 40;
pub const max_consts = 8;
pub const max_vars = 16;

pub const Op = enum(u8) {
    x, // variable
    c, // constant
    add,
    sub,
    mul,
    div,
    sin,
    cos,
    exp,
    log,
    sqrt,
    square,
    neg,

    pub fn arity(op: Op) u8 {
        return switch (op) {
            .x, .c => 0,
            .add, .sub, .mul, .div => 2,
            .sin, .cos, .exp, .log, .sqrt, .square, .neg => 1,
        };
    }
};

pub const Token = struct {
    op: Op,
    arg: u8 = 0, // variable or constant index
};

pub const Expr = struct {
    len: u8 = 0,
    nconst: u8 = 0,
    toks: [max_len]Token = undefined,
    consts: [max_consts]f64 = undefined,

    pub fn slice(e: *const Expr) []const Token {
        return e.toks[0..e.len];
    }

    /// Start index of the subtree that ends at token `end`.
    pub fn subtreeStart(e: *const Expr, end: usize) usize {
        var need: isize = 1;
        var j: isize = @intCast(end);
        while (true) {
            need += @as(isize, e.toks[@intCast(j)].op.arity()) - 1;
            if (need == 0) return @intCast(j);
            j -= 1;
        }
    }

    /// Number of constants actually referenced, after renumbering them in order
    /// of appearance (drops unused constants).
    pub fn compactConsts(e: *Expr) void {
        var map: [max_consts]u8 = @splat(255);
        var vals: [max_consts]f64 = undefined;
        var n: u8 = 0;
        for (e.toks[0..e.len]) |*t| {
            if (t.op != .c) continue;
            if (map[t.arg] == 255) {
                map[t.arg] = n;
                vals[n] = e.consts[t.arg];
                n += 1;
            }
            t.arg = map[t.arg];
        }
        e.nconst = n;
        @memcpy(e.consts[0..n], vals[0..n]);
    }

    pub fn eql(a: *const Expr, b: *const Expr) bool {
        if (a.len != b.len or a.nconst != b.nconst) return false;
        for (a.toks[0..a.len], b.toks[0..b.len]) |x, y| if (x.op != y.op or x.arg != y.arg) return false;
        for (a.consts[0..a.nconst], b.consts[0..b.nconst]) |x, y| if (x != y) return false;
        return true;
    }

    /// Writes the expression in Python syntax (x0, x1, ... for the variables,
    /// constants in full precision), e.g. "((x0 + 1.5) * sin(x1))".
    pub fn format(e: *const Expr, w: *std.Io.Writer) !void {
        var arena_buf: [16 * 1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&arena_buf);
        const a = fba.allocator();
        var stack: [max_len][]const u8 = undefined;
        var sp: usize = 0;
        for (e.toks[0..e.len]) |t| {
            const s: []const u8 = switch (t.op) {
                .x => try std.fmt.allocPrint(a, "x{d}", .{t.arg}),
                .c => try std.fmt.allocPrint(a, "{e}", .{e.consts[t.arg]}),
                .add, .sub, .mul, .div => blk: {
                    sp -= 1;
                    const sym = switch (t.op) {
                        .add => "+",
                        .sub => "-",
                        .mul => "*",
                        else => "/",
                    };
                    break :blk try std.fmt.allocPrint(a, "({s} {s} {s})", .{ stack[sp - 1], sym, stack[sp] });
                },
                .square => try std.fmt.allocPrint(a, "({s})**2", .{stack[sp - 1]}),
                .neg => try std.fmt.allocPrint(a, "(-{s})", .{stack[sp - 1]}),
                else => try std.fmt.allocPrint(a, "{s}({s})", .{ @tagName(t.op), stack[sp - 1] }),
            };
            if (t.op.arity() == 0) {
                stack[sp] = s;
                sp += 1;
            } else {
                stack[sp - 1] = s;
            }
        }
        try w.writeAll(stack[0]);
    }
};

/// The operators a search may use.
pub const OpSet = struct {
    binary: []const Op = &.{ .add, .sub, .mul, .div },
    unary: []const Op = &.{ .sin, .cos, .exp, .log, .sqrt, .square },
};

/// A random expression of at most `depth` levels, appended to `e`. Leaves are
/// variables (two thirds) or constants drawn from N(0, 1).
pub fn grow(e: *Expr, rand: std.Random, nvars: usize, ops: OpSet, depth: usize, full: bool) !void {
    const leaf = depth == 0 or (!full and rand.float(f64) < 0.3);
    if (leaf) {
        if (e.len >= max_len) return error.TooLong;
        if (e.nconst < max_consts and rand.float(f64) < 1.0 / 3.0) {
            e.consts[e.nconst] = rand.floatNorm(f64);
            e.toks[e.len] = .{ .op = .c, .arg = e.nconst };
            e.nconst += 1;
        } else {
            e.toks[e.len] = .{ .op = .x, .arg = @intCast(rand.uintLessThan(usize, nvars)) };
        }
        e.len += 1;
        return;
    }
    const total = ops.binary.len + ops.unary.len;
    const pick = rand.uintLessThan(usize, total);
    if (pick < ops.binary.len) {
        try grow(e, rand, nvars, ops, depth - 1, full);
        try grow(e, rand, nvars, ops, depth - 1, full);
        if (e.len >= max_len) return error.TooLong;
        e.toks[e.len] = .{ .op = ops.binary[pick] };
    } else {
        try grow(e, rand, nvars, ops, depth - 1, full);
        if (e.len >= max_len) return error.TooLong;
        e.toks[e.len] = .{ .op = ops.unary[pick - ops.binary.len] };
    }
    e.len += 1;
}

pub fn random(rand: std.Random, nvars: usize, ops: OpSet, max_depth: usize) Expr {
    while (true) {
        var e: Expr = .{};
        const depth = 1 + rand.uintLessThan(usize, max_depth);
        grow(&e, rand, nvars, ops, depth, rand.boolean()) catch continue;
        return e;
    }
}

/// Appends the tokens src.toks[from..to] to `out`, giving each constant a new
/// slot. False if `out` would be too long or use too many constants.
fn copyRange(out: *Expr, src: *const Expr, from: usize, to: usize) bool {
    for (src.toks[from..to]) |t| {
        if (out.len >= max_len) return false;
        var tt = t;
        if (t.op == .c) {
            if (out.nconst >= max_consts) return false;
            out.consts[out.nconst] = src.consts[t.arg];
            tt.arg = out.nconst;
            out.nconst += 1;
        }
        out.toks[out.len] = tt;
        out.len += 1;
    }
    return true;
}

/// Replaces the tokens [start, end] of `dst` with the tokens [s0, s1] of
/// `src`, carrying their constants. Fails if the result would be too long or
/// use too many constants.
pub fn splice(dst: *const Expr, start: usize, end: usize, src: *const Expr, s0: usize, s1: usize) ?Expr {
    var out: Expr = .{};
    if (!copyRange(&out, dst, 0, start)) return null;
    if (!copyRange(&out, src, s0, s1 + 1)) return null;
    if (!copyRange(&out, dst, end + 1, dst.len)) return null;
    return out;
}

/// Replaces the subtree [start, end] of `e`, call it s, by op(s, other) or,
/// if `other_first`, op(other, s), for a binary operator; by op(s) for a
/// unary one (`other` is then unused).
pub fn insert(e: *const Expr, start: usize, end: usize, other: *const Expr, op: Op, other_first: bool) ?Expr {
    var w: Expr = .{};
    if (op.arity() == 2 and other_first and !copyRange(&w, other, 0, other.len)) return null;
    if (!copyRange(&w, e, start, end + 1)) return null;
    if (op.arity() == 2 and !other_first and !copyRange(&w, other, 0, other.len)) return null;
    if (w.len >= max_len) return null;
    w.toks[w.len] = .{ .op = op };
    w.len += 1;
    return splice(e, start, end, &w, 0, w.len - 1);
}

/// The expression a + b * e, written with as few extra tokens as possible:
/// none when a = 0 and b = 1; a coefficient merged into a constant that the
/// root already multiplies, divides or adds; b = -1 as a negation or a
/// subtraction; otherwise "b mul" and "a add" appended. b = 0 gives the
/// constant a. Null if the result would be too long or use too many
/// constants.
pub fn materialise(e: *const Expr, a: f64, b: f64) ?Expr {
    var m = e.*;
    if (b == 0) {
        m = .{ .len = 1, .nconst = 1 };
        m.toks[0] = .{ .op = .c };
        m.consts[0] = a;
        return m;
    }
    var a_left = a;
    if (b == -1) {
        if (a != 0) {
            // a - e
            if (m.len + 2 > max_len or m.nconst >= max_consts) return null;
            var out: Expr = .{ .len = m.len + 2, .nconst = m.nconst + 1 };
            @memcpy(out.consts[0..m.nconst], m.consts[0..m.nconst]);
            out.consts[m.nconst] = a;
            out.toks[0] = .{ .op = .c, .arg = m.nconst };
            @memcpy(out.toks[1 .. m.len + 1], m.toks[0..m.len]);
            out.toks[m.len + 1] = .{ .op = .sub };
            out.compactConsts();
            return out;
        }
        if (m.len + 1 > max_len) return null;
        m.toks[m.len] = .{ .op = .neg };
        m.len += 1;
        return m;
    }
    if (b != 1) {
        const r = rootConsts(&m);
        if (r.op == .mul and (r.left != null or r.right != null)) {
            m.consts[(r.right orelse r.left).?] *= b;
        } else if (r.op == .div and r.left != null) {
            m.consts[r.left.?] *= b;
        } else if (r.op == .div and r.right != null) {
            m.consts[r.right.?] /= b;
        } else if (!append(&m, b, .mul)) return null;
    }
    if (a_left != 0) {
        const r = rootConsts(&m);
        if (r.op == .add and (r.left != null or r.right != null)) {
            m.consts[(r.right orelse r.left).?] += a_left;
        } else if (r.op == .sub and r.left != null) {
            m.consts[r.left.?] += a_left;
        } else if (r.op == .sub and r.right != null) {
            m.consts[r.right.?] -= a_left;
        } else if (!append(&m, a_left, .add)) return null;
        a_left = 0;
    }
    return m;
}

const RootConsts = struct { op: Op, left: ?u8 = null, right: ?u8 = null };

/// The root operator and, for a binary root, the constant index of each
/// child that is a single constant.
fn rootConsts(e: *const Expr) RootConsts {
    const root = e.toks[e.len - 1].op;
    if (root.arity() != 2) return .{ .op = root };
    const r_end = e.len - 2;
    const r_start = e.subtreeStart(r_end);
    var out: RootConsts = .{ .op = root };
    if (r_start == r_end and e.toks[r_end].op == .c) out.right = e.toks[r_end].arg;
    if (r_start == 1 and e.toks[0].op == .c) out.left = e.toks[0].arg;
    return out;
}

/// Appends a new constant `v` and the binary operator `op`.
fn append(e: *Expr, v: f64, op: Op) bool {
    if (e.len + 2 > max_len or e.nconst >= max_consts) return false;
    e.consts[e.nconst] = v;
    e.toks[e.len] = .{ .op = .c, .arg = e.nconst };
    e.toks[e.len + 1] = .{ .op = op };
    e.len += 2;
    e.nconst += 1;
    return true;
}

/// Folds every subtree made only of constants into a single constant, and
/// removes a few identities that add size without changing the value.
pub fn simplify(e: *Expr) void {
    var changed = true;
    while (changed) {
        changed = false;
        var i: usize = 0;
        while (i < e.len) : (i += 1) {
            const t = e.toks[i];
            if (t.op.arity() == 0) continue;
            const start = e.subtreeStart(i);
            // Constant subtree: evaluate it.
            var all_const = true;
            for (e.toks[start..i]) |u| {
                if (u.op == .x) {
                    all_const = false;
                    break;
                }
            }
            if (all_const and i > start) {
                const v = evalScalar(e, start, i);
                if (!std.math.isFinite(v)) continue;
                var c: Expr = .{ .len = 1, .nconst = 1 };
                c.toks[0] = .{ .op = .c, .arg = 0 };
                c.consts[0] = v;
                if (splice(e, start, i, &c, 0, 0)) |r| {
                    e.* = r;
                    changed = true;
                    break;
                }
            }
            // neg(neg(a)) -> a
            if (t.op == .neg and i > 0 and e.toks[i - 1].op == .neg) {
                const s = e.subtreeStart(i - 2);
                if (splice(e, start, i, e, s, i - 2)) |r| {
                    e.* = r;
                    changed = true;
                    break;
                }
            }
        }
    }
}

/// Value of the constant-only subtree [start, end].
fn evalScalar(e: *const Expr, start: usize, end: usize) f64 {
    var stack: [max_len]f64 = undefined;
    var sp: usize = 0;
    for (e.toks[start .. end + 1]) |t| {
        switch (t.op) {
            .c => {
                stack[sp] = e.consts[t.arg];
                sp += 1;
            },
            .x => return std.math.nan(f64),
            .add, .sub, .mul, .div => {
                sp -= 1;
                const b = stack[sp];
                const a = &stack[sp - 1];
                a.* = switch (t.op) {
                    .add => a.* + b,
                    .sub => a.* - b,
                    .mul => a.* * b,
                    else => a.* / b,
                };
            },
            else => stack[sp - 1] = unaryScalar(t.op, stack[sp - 1]),
        }
    }
    return stack[0];
}

pub fn unaryScalar(op: Op, v: f64) f64 {
    return switch (op) {
        .sin => @sin(v),
        .cos => @cos(v),
        .exp => @exp(v),
        .log => @log(v),
        .sqrt => @sqrt(v),
        .square => v * v,
        .neg => -v,
        else => unreachable,
    };
}

const testing = std.testing;

fn parseRpn(src: []const u8, consts: []const f64) Expr {
    var e: Expr = .{};
    var it = std.mem.tokenizeScalar(u8, src, ' ');
    while (it.next()) |w| {
        const t: Token = if (w[0] == 'x')
            .{ .op = .x, .arg = std.fmt.parseInt(u8, w[1..], 10) catch unreachable }
        else if (w[0] == 'c')
            .{ .op = .c, .arg = std.fmt.parseInt(u8, w[1..], 10) catch unreachable }
        else
            .{ .op = std.meta.stringToEnum(Op, w).? };
        e.toks[e.len] = t;
        e.len += 1;
    }
    e.nconst = @intCast(consts.len);
    @memcpy(e.consts[0..consts.len], consts);
    return e;
}

fn render(e: *const Expr, buf: []u8) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    e.format(&w) catch unreachable;
    return w.buffered();
}

test "subtrees, printing and splicing" {
    var buf: [256]u8 = undefined;
    const e = parseRpn("x0 c0 add x1 sin mul", &.{1.5});
    try testing.expectEqualStrings("((x0 + 1.5e0) * sin(x1))", render(&e, &buf));
    try testing.expectEqual(@as(usize, 0), e.subtreeStart(5));
    try testing.expectEqual(@as(usize, 3), e.subtreeStart(4));
    try testing.expectEqual(@as(usize, 0), e.subtreeStart(2));
    // Replace sin(x1) by x0 square.
    const f = parseRpn("x0 square", &.{});
    const g = splice(&e, 3, 4, &f, 0, 1).?;
    try testing.expectEqualStrings("((x0 + 1.5e0) * (x0)**2)", render(&g, &buf));
}

test "inserting an operator above a subtree" {
    var buf: [256]u8 = undefined;
    const e = parseRpn("x0 c0 add x1 sin mul", &.{1.5});
    const f = parseRpn("x2 c0 mul", &.{2});
    try testing.expectEqualStrings("(((x0 + 1.5e0) + (x2 * 2e0)) * sin(x1))", render(&insert(&e, 0, 2, &f, .add, false).?, &buf));
    try testing.expectEqualStrings("((x0 + 1.5e0) * ((x2 * 2e0) / sin(x1)))", render(&insert(&e, 3, 4, &f, .div, true).?, &buf));
    try testing.expectEqualStrings("exp(((x0 + 1.5e0) * sin(x1)))", render(&insert(&e, 0, 5, &f, .exp, false).?, &buf));
    const g = insert(&e, 0, 2, &f, .add, false).?;
    try testing.expectEqual(@as(u8, 2), g.nconst);
    try testing.expectEqual(@as(f64, 1.5), g.consts[0]);
    try testing.expectEqual(@as(f64, 2), g.consts[1]);
}

test "constant folding" {
    var buf: [256]u8 = undefined;
    var e = parseRpn("x0 c0 c1 mul add c2 neg neg mul", &.{ 2, 3, 0.5 });
    simplify(&e);
    try testing.expectEqualStrings("((x0 + 6e0) * 5e-1)", render(&e, &buf));
    try testing.expectEqual(@as(u8, 2), e.nconst);
}

test "writing a linear rescaling into an expression" {
    var buf: [256]u8 = undefined;
    const e = parseRpn("x0 sin", &.{});
    try testing.expectEqualStrings("sin(x0)", render(&materialise(&e, 0, 1).?, &buf));
    try testing.expectEqualStrings("((sin(x0) * 2e0) + 3e0)", render(&materialise(&e, 3, 2).?, &buf));
    try testing.expectEqualStrings("(-sin(x0))", render(&materialise(&e, 0, -1).?, &buf));
    try testing.expectEqualStrings("(4e0 - sin(x0))", render(&materialise(&e, 4, -1).?, &buf));
    try testing.expectEqualStrings("5e-1", render(&materialise(&e, 0.5, 0).?, &buf));
    // Coefficients merge into constants at the root.
    const f = parseRpn("c0 x0 mul", &.{3});
    try testing.expectEqualStrings("(6e0 * x0)", render(&materialise(&f, 0, 2).?, &buf));
    const g = parseRpn("x0 c0 div c1 add", &.{ 2, 1 });
    const h = materialise(&g, 1.5, 1).?;
    try testing.expectEqualStrings("((x0 / 2e0) + 2.5e0)", render(&h, &buf));
    try testing.expectEqual(g.len, h.len);
    const k = parseRpn("x0 c0 div", &.{2});
    try testing.expectEqualStrings("(x0 / 1e0)", render(&materialise(&k, 0, 2).?, &buf));
}

test "random expressions are well formed" {
    var prng = std.Random.DefaultPrng.init(1);
    for (0..1000) |_| {
        const e = random(prng.random(), 3, .{}, 5);
        var depth: isize = 0;
        for (e.slice()) |t| depth += 1 - @as(isize, t.op.arity());
        try testing.expectEqual(@as(isize, 1), depth);
        try testing.expectEqual(@as(usize, 0), e.subtreeStart(e.len - 1));
    }
}
