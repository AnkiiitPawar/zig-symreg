//! Symbolic regression by genetic programming: expressions in reverse Polish
//! notation, column-wise SIMD evaluation, Levenberg-Marquardt constant
//! fitting and a Pareto front of accuracy against size.

pub const expr = @import("expr.zig");
pub const eval = @import("eval.zig");
pub const fit = @import("fit.zig");
pub const search = @import("search.zig");

test {
    _ = expr;
    _ = eval;
    _ = fit;
    _ = search;
}
