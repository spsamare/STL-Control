"""STL formula objects and MILP basis encodings.

This module mirrors the reformulations in Section 4.1 of
"An STL-based Approach to Resilient Control for Cyber-Physical Systems".

The code is intentionally solver-neutral: formulas emit named variables and
linear constraints that can later be translated to Pyomo, Gurobi, or another
MILP backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


Number = int | float


@dataclass(frozen=True)
class Var:
    """Symbolic MILP variable reference."""

    name: str

    def expr(self) -> "LinExpr":
        return LinExpr({self.name: 1.0})

    def __add__(self, other: "LinearLike") -> "LinExpr":
        return self.expr() + other

    def __radd__(self, other: "LinearLike") -> "LinExpr":
        return self.expr() + other

    def __sub__(self, other: "LinearLike") -> "LinExpr":
        return self.expr() - other

    def __rsub__(self, other: "LinearLike") -> "LinExpr":
        return as_expr(other) - self.expr()

    def __mul__(self, scalar: Number) -> "LinExpr":
        return self.expr() * scalar

    def __rmul__(self, scalar: Number) -> "LinExpr":
        return self.expr() * scalar

    def __neg__(self) -> "LinExpr":
        return -self.expr()


@dataclass(frozen=True)
class LinExpr:
    """A linear expression: sum(coeff[name] * var[name]) + constant."""

    coeffs: Mapping[str, float] = field(default_factory=dict)
    constant: float = 0.0

    def __add__(self, other: LinearLike) -> "LinExpr":
        other_expr = as_expr(other)
        coeffs = dict(self.coeffs)
        for name, coeff in other_expr.coeffs.items():
            coeffs[name] = coeffs.get(name, 0.0) + coeff
            if coeffs[name] == 0:
                del coeffs[name]
        return LinExpr(coeffs, self.constant + other_expr.constant)

    def __radd__(self, other: LinearLike) -> "LinExpr":
        return self + other

    def __sub__(self, other: LinearLike) -> "LinExpr":
        return self + (-as_expr(other))

    def __rsub__(self, other: LinearLike) -> "LinExpr":
        return as_expr(other) - self

    def __mul__(self, scalar: Number) -> "LinExpr":
        return LinExpr(
            {name: coeff * float(scalar) for name, coeff in self.coeffs.items()},
            self.constant * float(scalar),
        )

    def __rmul__(self, scalar: Number) -> "LinExpr":
        return self * scalar

    def __neg__(self) -> "LinExpr":
        return self * -1


LinearLike = Number | Var | LinExpr


def as_expr(value: LinearLike) -> LinExpr:
    if isinstance(value, LinExpr):
        return value
    if isinstance(value, Var):
        return value.expr()
    return LinExpr({}, float(value))


@dataclass(frozen=True)
class Constraint:
    """Linear constraint lhs sense rhs."""

    lhs: LinExpr
    sense: str
    rhs: LinExpr
    label: str

    def normalized(self) -> LinExpr:
        """Return lhs - rhs, useful for backend adapters."""

        return self.lhs - self.rhs


@dataclass
class VariableSpec:
    name: str
    domain: str
    lb: Number | None = None
    ub: Number | None = None


class MILPEncoding:
    """Collects variables and constraints for STL/MILP reformulations."""

    def __init__(self, *, big_m: Number = 1_000_000):
        self.big_m = float(big_m)
        self.variables: dict[str, VariableSpec] = {}
        self.constraints: list[Constraint] = []
        self._ids: dict[str, int] = {}

    def fresh_name(self, prefix: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix)
        self._ids[clean] = self._ids.get(clean, 0) + 1
        return f"{clean}_{self._ids[clean]}"

    def var(
        self,
        name: str,
        *,
        domain: str,
        lb: Number | None = None,
        ub: Number | None = None,
    ) -> Var:
        if name in self.variables:
            return Var(name)
        self.variables[name] = VariableSpec(name=name, domain=domain, lb=lb, ub=ub)
        return Var(name)

    def binary(self, name: str) -> Var:
        return self.var(name, domain="binary", lb=0, ub=1)

    def integer(self, name: str, *, lb: Number = 0, ub: Number | None = None) -> Var:
        return self.var(name, domain="integer", lb=lb, ub=ub)

    def continuous(
        self, name: str, *, lb: Number | None = None, ub: Number | None = None
    ) -> Var:
        return self.var(name, domain="continuous", lb=lb, ub=ub)

    def add(self, lhs: LinearLike, sense: str, rhs: LinearLike, label: str) -> None:
        if sense not in {"<=", ">=", "=="}:
            raise ValueError(f"Unsupported constraint sense: {sense}")
        self.constraints.append(Constraint(as_expr(lhs), sense, as_expr(rhs), label))

    def add_eq(self, lhs: LinearLike, rhs: LinearLike, label: str) -> None:
        self.add(lhs, "==", rhs, label)

    def add_le(self, lhs: LinearLike, rhs: LinearLike, label: str) -> None:
        self.add(lhs, "<=", rhs, label)

    def add_ge(self, lhs: LinearLike, rhs: LinearLike, label: str) -> None:
        self.add(lhs, ">=", rhs, label)

    # Eq. (1): (z - 1) M <= mu(x_t) - c <= z M.
    def encode_predicate(
        self, z: Var, mu_minus_c: LinearLike, *, label: str = "eq1_predicate"
    ) -> None:
        expr = as_expr(mu_minus_c)
        self.add_ge(expr, (z - 1) * self.big_m, f"{label}_lower")
        self.add_le(expr, z * self.big_m, f"{label}_upper")

    # Eq. (2): z_not = 1 - z.
    def encode_not(self, out: Var, child: Var, *, label: str = "eq2_not") -> None:
        self.add_eq(out, 1 - child, label)

    # Eq. (3): out = and_i child_i.
    def encode_and(
        self, out: Var, children: Sequence[Var], *, label: str = "eq3_and"
    ) -> None:
        if not children:
            self.add_eq(out, 1, f"{label}_empty_true")
            return
        for i, child in enumerate(children):
            self.add_le(out, child, f"{label}_upper_{i}")
        self.add_ge(out, 1 - len(children) + sum_expr(children), f"{label}_lower")

    # Eq. (4): out = or_i child_i.
    def encode_or(
        self, out: Var, children: Sequence[Var], *, label: str = "eq4_or"
    ) -> None:
        if not children:
            self.add_eq(out, 0, f"{label}_empty_false")
            return
        for i, child in enumerate(children):
            self.add_ge(out, child, f"{label}_lower_{i}")
        self.add_le(out, sum_expr(children), f"{label}_upper")

    # Eq. (9): y = z * c for binary z and bounded integer/linear c.
    def encode_binary_product(
        self,
        y: Var,
        z: Var,
        c: LinearLike,
        *,
        lower_bound: Number,
        upper_bound: Number,
        label: str = "eq9_binary_product",
    ) -> None:
        m = float(lower_bound)
        big_m = float(upper_bound)
        c_expr = as_expr(c)
        self.add_ge(y, m * z, f"{label}_lower_active")
        self.add_le(y, big_m * z, f"{label}_upper_active")
        self.add_ge(y, c_expr - big_m * (1 - z), f"{label}_lower_value")
        self.add_le(y, c_expr - m * (1 - z), f"{label}_upper_value")


def sum_expr(values: Iterable[LinearLike]) -> LinExpr:
    total = LinExpr()
    for value in values:
        total += value
    return total


@dataclass(frozen=True)
class EncodingResult:
    z: dict[int, Var]

    def at(self, t: int) -> Var:
        return self.z[t]


class Formula:
    """Base class for STL formulas."""

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        raise NotImplementedError

    def z_name(self, model: MILPEncoding, t: int) -> str:
        return model.fresh_name(f"z_{self.kind}_{t}")

    @property
    def kind(self) -> str:
        return self.__class__.__name__.lower()

    def __invert__(self) -> "Not":
        return Not(self)

    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)


@dataclass(frozen=True)
class Predicate(Formula):
    """Atomic proposition p == mu(x_t) >= c.

    `mu_minus_c(t, model)` must return the linear expression mu(x_t) - c.
    For example, for x_0[t] <= xmax, encode as Predicate(lambda t, m:
    xmax - m.continuous(f"x_{t}_0"), name="x0_le_xmax").
    """

    mu_minus_c: Callable[[int, MILPEncoding], LinearLike]
    name: str = "predicate"

    @property
    def kind(self) -> str:
        return self.name

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            z[t] = model.binary(f"z_{self.name}_{t}")
            model.encode_predicate(
                z[t],
                self.mu_minus_c(t, model),
                label=f"eq1_{self.name}_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class Not(Formula):
    child: Formula

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        child = self.child.encode(model, horizon)
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            z[t] = model.binary(self.z_name(model, t))
            model.encode_not(z[t], child.at(t), label=f"eq2_not_{t}")
        return EncodingResult(z)


@dataclass(frozen=True)
class And(Formula):
    children: tuple[Formula, ...]

    def __init__(self, *children: Formula):
        object.__setattr__(self, "children", tuple(children))

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        encoded = [child.encode(model, horizon) for child in self.children]
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            z[t] = model.binary(self.z_name(model, t))
            model.encode_and(
                z[t],
                [child.at(t) for child in encoded],
                label=f"eq3_and_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class Or(Formula):
    children: tuple[Formula, ...]

    def __init__(self, *children: Formula):
        object.__setattr__(self, "children", tuple(children))

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        encoded = [child.encode(model, horizon) for child in self.children]
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            z[t] = model.binary(self.z_name(model, t))
            model.encode_or(
                z[t],
                [child.at(t) for child in encoded],
                label=f"eq4_or_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class Always(Formula):
    a: int
    b: int
    child: Formula

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        check_interval(self.a, self.b)
        child = self.child.encode(model, horizon)
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            start = min(self.a + t, horizon)
            stop = min(self.b + t, horizon)
            z[t] = model.binary(self.z_name(model, t))
            model.encode_and(
                z[t],
                [child.at(i) for i in range(start, stop + 1)],
                label=f"always_{self.a}_{self.b}_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class Eventually(Formula):
    a: int
    b: int
    child: Formula

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        check_interval(self.a, self.b)
        child = self.child.encode(model, horizon)
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            start = min(self.a + t, horizon)
            stop = min(self.b + t, horizon)
            z[t] = model.binary(self.z_name(model, t))
            model.encode_or(
                z[t],
                [child.at(i) for i in range(start, stop + 1)],
                label=f"eventually_{self.a}_{self.b}_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class Until(Formula):
    a: int
    b: int
    left: Formula
    right: Formula

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        check_interval(self.a, self.b)
        if self.a == 0:
            phi = And(
                Eventually(0, self.b, self.right),
                Eventually(0, 0, UnboundedUntil(self.left, self.right)),
            )
        else:
            phi = And(
                Always(0, self.a - 1, self.left),
                Eventually(self.a, self.b, self.right),
                Eventually(self.a, self.a, UnboundedUntil(self.left, self.right)),
            )
        return phi.encode(model, horizon)


@dataclass(frozen=True)
class UnboundedUntil(Formula):
    """Auxiliary formula for Eq. (5)'s unbounded until recurrence."""

    left: Formula
    right: Formula

    def encode(self, model: MILPEncoding, horizon: int) -> EncodingResult:
        left = self.left.encode(model, horizon)
        right = self.right.encode(model, horizon)
        z: dict[int, Var] = {}
        for t in range(horizon + 1):
            z[t] = model.binary(self.z_name(model, t))

        model.add_eq(z[horizon], right.at(horizon), f"until_unbounded_terminal_{horizon}")
        for t in range(horizon - 1, -1, -1):
            left_and_next = model.binary(model.fresh_name(f"z_until_and_{t}"))
            model.encode_and(
                left_and_next,
                [left.at(t), z[t + 1]],
                label=f"until_unbounded_and_{t}",
            )
            model.encode_or(
                z[t],
                [right.at(t), left_and_next],
                label=f"until_unbounded_or_{t}",
            )
        return EncodingResult(z)


@dataclass(frozen=True)
class ResilienceEncodingResult:
    z_phi: dict[int, Var]
    c_rec: dict[int, Var]
    c_dur: dict[int, Var]
    c1: dict[int, Var]
    c2: dict[int, Var]

    @property
    def t_rec(self) -> Var:
        return self.c_rec[0]

    @property
    def t_dur(self) -> Var:
        return self.c_dur[0]


def encode_resilience_counters(
    phi: Formula,
    model: MILPEncoding,
    horizon: int,
    *,
    prefix: str = "phi",
) -> ResilienceEncodingResult:
    """Encode Eqs. (6)-(8) for t_rec(phi, x, t) and t_dur(phi, x, t).

    The nonlinear binary-counter products are linearized with Eq. (9).
    Bounds use [0, horizon], which is tight for the finite-horizon counters.
    """

    encoded_phi = phi.encode(model, horizon)
    z = encoded_phi.z

    c_rec = {
        t: model.integer(f"c_{prefix}_rec_{t}", lb=0, ub=horizon)
        for t in range(horizon + 1)
    }
    c1 = {
        t: model.integer(f"c_{prefix}_dur_true_{t}", lb=0, ub=horizon)
        for t in range(horizon + 1)
    }
    c2 = {
        t: model.integer(f"c_{prefix}_dur_after_recovery_{t}", lb=0, ub=horizon)
        for t in range(horizon + 1)
    }
    c_dur = {
        t: model.integer(f"c_{prefix}_dur_{t}", lb=0, ub=horizon)
        for t in range(horizon + 1)
    }

    model.add_eq(c_rec[horizon], 0, f"eq6_rec_terminal_{horizon}")
    model.add_eq(c1[horizon], 0, f"eq7_c1_terminal_{horizon}")
    model.add_eq(c2[horizon], 0, f"eq7_c2_terminal_{horizon}")
    model.add_eq(c_dur[horizon], c1[horizon] + c2[horizon], f"eq8_dur_terminal_{horizon}")

    for t in range(horizon - 1, -1, -1):
        not_z = model.binary(model.fresh_name(f"z_not_{prefix}_{t}"))
        model.encode_not(not_z, z[t], label=f"res_not_z_{t}")

        # Eq. (6): c_rec[t] = (1 - z_phi[t]) * (c_rec[t + 1] + 1).
        model.encode_binary_product(
            c_rec[t],
            not_z,
            c_rec[t + 1] + 1,
            lower_bound=0,
            upper_bound=horizon,
            label=f"eq6_rec_{t}",
        )

        # Eq. (7): c1[t] = z_phi[t] * (c1[t + 1] + 1).
        model.encode_binary_product(
            c1[t],
            z[t],
            c1[t + 1] + 1,
            lower_bound=0,
            upper_bound=horizon,
            label=f"eq7_c1_{t}",
        )

        # Eq. (7): c2[t] = (1 - z_phi[t]) * (c1[t + 1] + c2[t + 1]).
        model.encode_binary_product(
            c2[t],
            not_z,
            c1[t + 1] + c2[t + 1],
            lower_bound=0,
            upper_bound=horizon,
            label=f"eq7_c2_{t}",
        )

        # Eq. (8): c_dur[t] = c1[t] + c2[t].
        model.add_eq(c_dur[t], c1[t] + c2[t], f"eq8_dur_{t}")

    return ResilienceEncodingResult(
        z_phi=z,
        c_rec=c_rec,
        c_dur=c_dur,
        c1=c1,
        c2=c2,
    )


def recoverability_objective(alpha: int, counters: ResilienceEncodingResult) -> LinExpr:
    """Return alpha - t_rec(phi, x, 0), the paper's rec objective."""

    return alpha - counters.t_rec


def durability_objective(beta: int, counters: ResilienceEncodingResult) -> LinExpr:
    """Return t_dur(phi, x, 0) - beta, the paper's dur objective."""

    return counters.t_dur - beta


def check_interval(a: int, b: int) -> None:
    if a < 0 or b < 0:
        raise ValueError("STL intervals must be non-negative.")
    if a > b:
        raise ValueError("STL interval lower bound must be <= upper bound.")
