//! zsr: symbolic regression on a CSV file.
//!
//!   zsr data.csv [--time 30] [--seed 1] [--islands 4] [--population 250]
//!                [--threads 1] [--generations N] [--migrate-every 50]
//!                [--scaling on|off] [--parsimony 0.03[,...]] [--insert 0.1]
//!                [--ops add,sub,mul,div,sin,cos,exp,log,sqrt,square]
//!                [--json out.json]
//!
//! The first line of the CSV names the columns; the last column is the target.

const std = @import("std");
const lib = @import("symreg");
const expr = lib.expr;
const eval = lib.eval;
const search = lib.search;

const usage =
    \\usage: zsr data.csv [--time 30] [--seed 1] [--islands 4] [--population 250]
    \\                    [--threads 1] [--generations N] [--migrate-every 50]
    \\                    [--scaling on|off] [--parsimony 0.03[,...]] [--insert 0.1]
    \\                    [--ops add,sub,mul,div,...] [--json out.json]
    \\The first line names the columns; the last column is the target.
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) fail(io, "missing data file");

    var cfg: search.Config = .{};
    var json_path: ?[]const u8 = null;
    var binary: std.ArrayList(expr.Op) = .empty;
    var unary: std.ArrayList(expr.Op) = .empty;
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) fail(io, "option needs a value");
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--time")) {
            cfg.time_limit = std.fmt.parseFloat(f64, v) catch fail(io, "bad --time");
        } else if (std.mem.eql(u8, k, "--seed")) {
            cfg.seed = std.fmt.parseInt(u64, v, 10) catch fail(io, "bad --seed");
        } else if (std.mem.eql(u8, k, "--islands")) {
            cfg.islands = std.fmt.parseInt(usize, v, 10) catch fail(io, "bad --islands");
        } else if (std.mem.eql(u8, k, "--population")) {
            cfg.population = std.fmt.parseInt(usize, v, 10) catch fail(io, "bad --population");
        } else if (std.mem.eql(u8, k, "--threads")) {
            cfg.threads = std.fmt.parseInt(usize, v, 10) catch fail(io, "bad --threads");
        } else if (std.mem.eql(u8, k, "--generations")) {
            cfg.max_generations = std.fmt.parseInt(usize, v, 10) catch fail(io, "bad --generations");
            cfg.time_limit = std.math.inf(f64);
        } else if (std.mem.eql(u8, k, "--migrate-every")) {
            cfg.migrate_every = std.fmt.parseInt(usize, v, 10) catch fail(io, "bad --migrate-every");
            if (cfg.migrate_every == 0) fail(io, "bad --migrate-every");
        } else if (std.mem.eql(u8, k, "--insert")) {
            cfg.p_insert = std.fmt.parseFloat(f64, v) catch fail(io, "bad --insert");
        } else if (std.mem.eql(u8, k, "--parsimony")) {
            var list: std.ArrayList(f64) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |p| try list.append(arena, std.fmt.parseFloat(f64, p) catch fail(io, "bad --parsimony"));
            cfg.parsimony = list.items;
        } else if (std.mem.eql(u8, k, "--scaling")) {
            if (std.mem.eql(u8, v, "on")) cfg.linear_scaling = true else if (std.mem.eql(u8, v, "off")) cfg.linear_scaling = false else fail(io, "bad --scaling");
        } else if (std.mem.eql(u8, k, "--json")) {
            json_path = v;
        } else if (std.mem.eql(u8, k, "--ops")) {
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |name| {
                const op = std.meta.stringToEnum(expr.Op, name) orelse fail(io, "unknown operator");
                if (op.arity() == 2) try binary.append(arena, op) else if (op.arity() == 1) try unary.append(arena, op) else fail(io, "not an operator");
            }
        } else fail(io, "unknown option");
    }
    if (binary.items.len > 0) cfg.ops.binary = binary.items;
    if (unary.items.len > 0 or binary.items.len > 0) cfg.ops.unary = unary.items;
    if (cfg.ops.binary.len == 0) fail(io, "need at least one binary operator");

    // Read the CSV.
    const text = std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .unlimited) catch fail(io, "cannot read the data file");
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    const header = lines.next() orelse fail(io, "empty file");
    var names: std.ArrayList([]const u8) = .empty;
    var hit = std.mem.splitScalar(u8, header, ',');
    while (hit.next()) |nm| try names.append(arena, std.mem.trim(u8, nm, " "));
    const ncols = names.items.len;
    if (ncols < 2 or ncols - 1 > expr.max_vars) fail(io, "need between 1 and 16 input columns");
    var cols = try arena.alloc(std.ArrayList(f64), ncols);
    for (cols) |*c| c.* = .empty;
    while (lines.next()) |line| {
        var it = std.mem.splitScalar(u8, line, ',');
        for (cols) |*c| {
            const f = it.next() orelse fail(io, "short row");
            try c.append(arena, std.fmt.parseFloat(f64, std.mem.trim(u8, f, " ")) catch fail(io, "bad number"));
        }
    }
    const xs = try arena.alloc([]const f64, ncols - 1);
    for (xs, cols[0 .. ncols - 1]) |*x, c| x.* = c.items;
    const d = try eval.Dataset.init(arena, xs, cols[ncols - 1].items);

    const res = try search.run(gpa, io, d, cfg);
    defer gpa.free(res.front);

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &stdout.interface;
    try w.print("{d} generations, {d} evaluations and {d} constant fits ({d} evaluations) in {d:.1} s\nPareto front (size, NMSE, expression):\n", .{ res.generations, res.evaluations, res.lm_calls, res.lm_evaluations, res.seconds });
    for (res.front, 0..) |f, j| {
        try w.print("{s} {d:>3}  {e:<12.3}  ", .{ if (j == res.selected) "*" else " ", f.e.len, f.loss });
        try f.e.format(w);
        try w.writeAll("\n");
    }
    try w.print("variables: ", .{});
    for (names.items[0 .. ncols - 1], 0..) |nm, j| try w.print("{s}x{d} = {s}", .{ if (j > 0) ", " else "", j, nm });
    try w.writeAll("\n");
    try w.flush();

    if (json_path) |p| {
        const file = try std.Io.Dir.cwd().createFile(io, p, .{});
        defer file.close(io);
        var fbuf: [64 * 1024]u8 = undefined;
        var fw = file.writerStreaming(io, &fbuf);
        const o = &fw.interface;
        try o.print("{{\"generations\": {d}, \"evaluations\": {d}, \"lm_calls\": {d}, \"lm_evaluations\": {d}, \"seconds\": {d}, \"exact_at\": ", .{ res.generations, res.evaluations, res.lm_calls, res.lm_evaluations, res.seconds });
        if (res.exact_at) |t| try o.print("{d}", .{t}) else try o.writeAll("null");
        try o.print(", \"selected\": {d}, \"front\": [", .{res.selected});
        for (res.front, 0..) |f, j| {
            try o.print("{s}{{\"size\": {d}, \"nmse\": {e}, \"expr\": \"", .{ if (j > 0) ", " else "", f.e.len, f.loss });
            try f.e.format(o);
            try o.writeAll("\"}");
        }
        try o.writeAll("]}\n");
        try o.flush();
    }
}

fn fail(io: std.Io, msg: []const u8) noreturn {
    var buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &buf);
    stderr.interface.print("zsr: {s}\n{s}", .{ msg, usage }) catch {};
    stderr.interface.flush() catch {};
    std.process.exit(1);
}
