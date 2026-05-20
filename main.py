"""Basic diagnostic entry point for the STL/MILP encoding layer.

Run this file after creating your Conda environment. It does not solve an MILP;
it only builds a small formula and prints the symbolic variables/constraints so
you can cross-check them against the paper's reformulations.
"""

from formula import (
    Always,
    MILPEncoding,
    Predicate,
    durability_objective,
    encode_resilience_counters,
    recoverability_objective,
)


def format_expr(expr):
    """Pretty-print the small symbolic linear expressions used by formula.py."""

    terms = []
    for name, coeff in sorted(expr.coeffs.items()):
        terms.append(f"{coeff:g}*{name}")
    if expr.constant:
        terms.append(f"{expr.constant:g}")
    return " + ".join(terms) if terms else "0"


def format_constraint(constraint):
    return (
        f"{constraint.label}: "
        f"{format_expr(constraint.lhs)} {constraint.sense} {format_expr(constraint.rhs)}"
    )


def build_demo_encoding():
    horizon = 5
    alpha = 2
    beta = 2
    big_m = 1000

    model = MILPEncoding(big_m=big_m)

    # Example atomic predicate:
    # p(t) := x_0(t) >= 0
    #
    # The state variable is declared here as a symbolic continuous variable.
    # Later, a solver adapter will connect x_0(t) to system dynamics.
    safe = Predicate(
        lambda t, m: m.continuous(f"x_{t}_0", lb=-10, ub=10),
        name="x0_ge_0",
    )

    # Example STL property:
    # phi := G_[0,2] p
    phi = Always(0, 2, safe)

    counters = encode_resilience_counters(
        phi,
        model,
        horizon,
        prefix="safe",
    )

    rec = recoverability_objective(alpha, counters)
    dur = durability_objective(beta, counters)

    return model, rec, dur


def main():
    model, rec, dur = build_demo_encoding()

    print("STL/MILP encoding demo")
    print(f"Variables:   {len(model.variables)}")
    print(f"Constraints: {len(model.constraints)}")
    print()
    print(f"rec objective: {format_expr(rec)}")
    print(f"dur objective: {format_expr(dur)}")
    print()
    print("First 12 constraints:")
    for constraint in model.constraints[:12]:
        print(f"  {format_constraint(constraint)}")


if __name__ == "__main__":
    main()
