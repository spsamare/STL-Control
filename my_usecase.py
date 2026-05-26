"""Control and sensing decisions for the noisy linear plant simulation.

At each step, the optimization gives:
    u: continuous control input

The sensing decision is determined by an age-of-information rule. CVXPY
solves a mixed-integer plan with a resilience-constrained STL state predicate.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


def sensing_age_limit(a, sigma, noise_threshold):
    """Compute the age limit used to trigger a new sensor observation.

    limit = logdet(I - s/sigma * (I - A A.T)) / logdet(A A.T)
    """

    a = np.asarray(a, dtype=float)
    sigma = float(sigma)
    noise_threshold = float(noise_threshold)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square state-transition matrix.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if noise_threshold < 0:
        raise ValueError("noise_threshold must be non-negative.")

    identity = np.eye(a.shape[0])
    aat = a @ a.T
    trigger_matrix = identity - (noise_threshold / sigma) * (identity - aat)
    denominator_sign, denominator = np.linalg.slogdet(aat)
    numerator_sign, numerator = np.linalg.slogdet(trigger_matrix)

    if denominator_sign <= 0 or numerator_sign <= 0:
        raise ValueError(
            "The logdet sensing threshold is undefined for the provided "
            "a, sigma, and noise_threshold."
        )
    if np.isclose(denominator, 0.0):
        raise ValueError("logdet(a @ a.T) must be nonzero.")

    return numerator / denominator


@dataclass(frozen=True)
class SensingDecision:
    time_index: int
    age_before_decision: int
    age_limit: float
    sense: bool
    age_of_information: int


class AgeOfInformationSensingPolicy:
    """Schedule sensing from an observable age-of-information counter.

    At t=0, a sensing action is forced. Thereafter, sensing is performed if
    the observable counter exceeds the configured logdet age limit.
    """

    def __init__(self, *, a, sigma, noise_threshold):
        self.a = np.asarray(a, dtype=float)
        self.sigma = float(sigma)
        self.noise_threshold = float(noise_threshold)
        self.reset()

    @property
    def age_limit(self):
        return sensing_age_limit(self.a, self.sigma, self.noise_threshold)

    @property
    def sensing_schedule(self):
        return [int(decision.sense) for decision in self.decision_history]

    def reset(self):
        self.age_of_information = 1
        self.time_index = 0
        self.decision_history: list[SensingDecision] = []

    def set_sigma(self, sigma):
        """Update process-noise variance used by future sensing decisions."""

        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = sigma

    def decide(self):
        """Return and record the sensing decision for the current time slot."""

        age_before = self.age_of_information
        age_without_sensing = age_before + 1
        should_sense = (
            self.time_index == 0 or age_without_sensing > self.age_limit
        )
        self.age_of_information = 1 if should_sense else age_without_sensing

        decision = SensingDecision(
            time_index=self.time_index,
            age_before_decision=age_before,
            age_limit=self.age_limit,
            sense=should_sense,
            age_of_information=self.age_of_information,
        )
        self.decision_history.append(decision)
        self.time_index += 1
        return decision


@dataclass(frozen=True)
class ControlPlan:
    start_time: int
    observed_state: np.ndarray
    estimated_states: np.ndarray
    controls: np.ndarray
    objective: float
    predicate_satisfaction: np.ndarray
    recovery_time: int
    durability_time: int
    robust_margins: np.ndarray


@dataclass(frozen=True)
class STLResilienceConfig:
    """Parameters for R_(alpha,beta)(||x - target|| <= delta)."""

    delta: float
    alpha: int
    beta: int
    epsilon: float
    process_noise_variance: float | np.ndarray
    big_m: float = 1_000.0
    norm_type: str = "infinity"
    solver: str = "SCIP"


@dataclass(frozen=True)
class ControlDecision:
    estimated_state: np.ndarray
    control: np.ndarray
    plan: ControlPlan
    planned_from_observation: bool

    @property
    def estimated_states(self):
        return self.plan.estimated_states

    @property
    def controls(self):
        return self.plan.controls


def solve_control_decision(
    *,
    observed_state,
    sensing_decision,
    target_state,
    control_coefficient_matrix,
    tx_power,
    stl_config,
    a,
    b,
    control_lower_bound=None,
    control_upper_bound=None,
    last_plan=None,
):
    """Create or consume a control plan according to the sensing decision.

    When a state is sensed, a finite-horizon mixed-integer quadratic program
    estimates future states and controls and records the transmission-power
    cost. Without sensing, the relevant cached state estimate and control are
    returned without charging transmission power again.
    """

    if sensing_decision.sense:
        horizon = max(1, int(np.floor(sensing_decision.age_limit)))
        plan = build_control_plan(
            start_time=sensing_decision.time_index,
            observed_state=observed_state,
            target_state=target_state,
            control_coefficient_matrix=control_coefficient_matrix,
            tx_power=tx_power,
            stl_config=stl_config,
            a=a,
            b=b,
            control_lower_bound=control_lower_bound,
            control_upper_bound=control_upper_bound,
            horizon=horizon,
        )
        return ControlDecision(
            estimated_state=plan.observed_state.copy(),
            control=plan.controls[0].copy(),
            plan=plan,
            planned_from_observation=True,
        )

    if last_plan is None:
        raise ValueError("A previous control plan is required when no state is sensed.")

    plan_offset = sensing_decision.time_index - last_plan.start_time
    if plan_offset < 1 or plan_offset >= len(last_plan.controls):
        raise ValueError(
            "No cached control is available at the current time; "
            "a new sensing decision is required."
        )

    return ControlDecision(
        estimated_state=last_plan.estimated_states[plan_offset - 1].copy(),
        control=last_plan.controls[plan_offset].copy(),
        plan=last_plan,
        planned_from_observation=False,
    )


def build_control_plan(
    *,
    start_time,
    observed_state,
    target_state,
    control_coefficient_matrix,
    tx_power,
    stl_config,
    a,
    b,
    control_lower_bound=None,
    control_upper_bound=None,
    horizon,
):
    """Solve a sensed-state resilient STL plan and include that event's tx power.

    The objective includes only control effort and the current sensing
    transmission power. State deviation is imposed by resilient STL
    constraints rather than penalized in the objective.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed_state = np.asarray(observed_state, dtype=float).reshape(-1)
    target_state = np.asarray(target_state, dtype=float).reshape(-1)
    r = np.asarray(control_coefficient_matrix, dtype=float)

    validate_control_problem(
        a,
        b,
        observed_state,
        target_state,
        r,
        tx_power,
        stl_config,
        horizon,
    )

    state_dim = a.shape[0]
    control_dim = b.shape[1]
    states = cp.Variable((horizon + 1, state_dim), name="estimated_states")
    controls = cp.Variable((horizon, control_dim), name="controls")
    z_phi = cp.Variable(horizon, boolean=True, name="z_phi")
    recovery = cp.Variable(horizon + 1, integer=True, name="recovery")
    dur_true = cp.Variable(horizon + 1, integer=True, name="dur_true")
    dur_after_recovery = cp.Variable(
        horizon + 1, integer=True, name="dur_after_recovery"
    )

    margins = robust_state_margins(
        a,
        stl_config.process_noise_variance,
        horizon,
        stl_config.epsilon,
        stl_config.norm_type,
    )
    constraints = [states[0] == observed_state]
    if control_lower_bound is not None:
        constraints.append(controls >= np.asarray(control_lower_bound, dtype=float))
    if control_upper_bound is not None:
        constraints.append(controls <= np.asarray(control_upper_bound, dtype=float))
    for k in range(horizon):
        constraints.append(states[k + 1] == a @ states[k] + b @ controls[k])
        constraints.extend(
            robust_predicate_constraints(
                states[k + 1],
                target_state,
                z_phi[k],
                margins[k],
                stl_config,
            )
        )

    constraints.extend(
        resilience_counter_constraints(
            z_phi,
            recovery,
            dur_true,
            dur_after_recovery,
            stl_config.alpha,
            stl_config.beta,
        )
    )

    objective = cp.Minimize(
        cp.sum([cp.quad_form(controls[k], r) for k in range(horizon)])
        + float(tx_power)
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=stl_config.solver)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"No feasible resilient STL control plan found: {problem.status}")

    estimated_states = np.asarray(states.value[1:], dtype=float)
    control_values = np.asarray(controls.value, dtype=float)
    z_values = np.rint(np.asarray(z_phi.value, dtype=float)).astype(int)

    return ControlPlan(
        start_time=start_time,
        observed_state=observed_state.copy(),
        estimated_states=estimated_states,
        controls=control_values,
        objective=float(problem.value),
        predicate_satisfaction=z_values,
        recovery_time=int(round(float(recovery.value[0]))),
        durability_time=int(round(float(dur_true.value[0] + dur_after_recovery.value[0]))),
        robust_margins=margins,
    )


def quadratic_control_cost(controls, coefficient_matrix):
    return sum(float(control.T @ coefficient_matrix @ control) for control in controls)


def robust_state_margins(a, process_noise_variance, horizon, epsilon, norm_type):
    """Moment-based chance tightening with epsilon shared across the horizon.

    A union-bound allocation of epsilon / horizon is used for each encoded
    predicate slot. This is a conservative sufficient condition for keeping
    the resilient trajectory violation probability within epsilon.
    """

    state_dim = a.shape[0]
    covariance = noise_covariance(process_noise_variance, state_dim)
    predicted_covariance = np.zeros((state_dim, state_dim))
    per_step_epsilon = epsilon / horizon
    margins = []
    for _ in range(horizon):
        predicted_covariance = a @ predicted_covariance @ a.T + covariance
        if norm_type == "l2":
            margins.append(np.sqrt(np.trace(predicted_covariance) / per_step_epsilon))
        elif norm_type == "infinity":
            margins.append(
                np.sqrt(state_dim * np.diag(predicted_covariance) / per_step_epsilon)
            )
        else:
            raise ValueError("norm_type must be 'l2' or 'infinity'.")
    return np.asarray(margins)


def robust_predicate_constraints(state, target_state, z_phi, margin, config):
    """Encode z_phi -> robust satisfaction of the state-deviation predicate."""

    relaxed_radius = config.delta + config.big_m * (1 - z_phi)
    if config.norm_type == "l2":
        return [cp.norm(state - target_state, 2) + float(margin) <= relaxed_radius]

    return [
        cp.abs(state[j] - target_state[j]) + float(margin[j]) <= relaxed_radius
        for j in range(len(target_state))
    ]


def resilience_counter_constraints(z_phi, recovery, dur_true, dur_after, alpha, beta):
    """MILP recurrences for R_(alpha,beta)(phi), following formula.py."""

    horizon = z_phi.shape[0]
    constraints = [
        recovery[horizon] == 0,
        dur_true[horizon] == 0,
        dur_after[horizon] == 0,
        recovery >= 0,
        dur_true >= 0,
        dur_after >= 0,
    ]
    for k in range(horizon):
        upper = horizon - k
        constraints.extend(
            [
                recovery[k] <= upper,
                dur_true[k] <= upper,
                dur_after[k] <= upper,
                dur_true[k] + dur_after[k] <= upper,
            ]
        )

    for k in range(horizon - 1, -1, -1):
        upper = horizon - k
        constraints.extend(
            binary_product_constraints(recovery[k], 1 - z_phi[k], recovery[k + 1] + 1, upper)
        )
        constraints.extend(
            binary_product_constraints(dur_true[k], z_phi[k], dur_true[k + 1] + 1, upper)
        )
        constraints.extend(
            binary_product_constraints(
                dur_after[k], 1 - z_phi[k], dur_true[k + 1] + dur_after[k + 1], upper
            )
        )

    constraints.extend([recovery[0] <= alpha, dur_true[0] + dur_after[0] >= beta])
    return constraints


def binary_product_constraints(product, binary, value, upper):
    """Linearize product == binary * value for nonnegative bounded value."""

    return [
        product <= upper * binary,
        product <= value,
        product >= value - upper * (1 - binary),
    ]


def noise_covariance(variance, state_dim):
    variance = np.asarray(variance, dtype=float)
    if variance.shape == ():
        return np.eye(state_dim) * float(variance)
    if variance.ndim == 1:
        return np.diag(variance)
    return variance


def validate_control_problem(a, b, observed_state, target_state, r, tx_power, config, horizon):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square state-transition matrix.")
    if b.ndim != 2 or b.shape[0] != a.shape[0]:
        raise ValueError("b must have one row per state dimension.")
    if observed_state.size != a.shape[0] or target_state.size != a.shape[0]:
        raise ValueError("observed_state and target_state must match the state size.")
    if r.shape != (b.shape[1], b.shape[1]):
        raise ValueError("control_coefficient_matrix must match the control size.")
    if float(tx_power) < 0:
        raise ValueError("tx_power must be non-negative.")
    if config.delta <= 0:
        raise ValueError("STL deviation radius delta must be positive.")
    if config.alpha < 0 or config.beta < 0:
        raise ValueError("STL alpha and beta must be non-negative.")
    if not 0 < config.epsilon < 1:
        raise ValueError("STL chance epsilon must lie strictly between 0 and 1.")
    if config.alpha >= horizon or config.beta > horizon:
        raise ValueError("STL alpha and beta must fit within the prediction horizon.")
