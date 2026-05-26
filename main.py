"""Entry point for the configurable resilient-control simulation."""

import numpy as np

import simulation_config as config
from formula import (
    Always,
    MILPEncoding,
    Predicate,
    durability_objective,
    encode_resilience_counters,
    recoverability_objective,
)
from my_usecase import (
    AgeOfInformationSensingPolicy,
    STLResilienceConfig,
    solve_control_decision,
)
from plant import NoisyLinearPlant
from visualization import plot_plant_history


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
    horizon = config.ENCODING_HORIZON

    model = MILPEncoding(big_m=config.ENCODING_BIG_M)

    # Example atomic predicate:
    # p(t) := x_0(t) >= 0
    #
    # The state variable is declared here as a symbolic continuous variable.
    # Later, a solver adapter will connect x_0(t) to system dynamics.
    safe = Predicate(
        lambda t, m: m.continuous(
            f"x_{t}_0",
            lb=config.ENCODING_STATE_LOWER_BOUND,
            ub=config.ENCODING_STATE_UPPER_BOUND,
        ),
        name="x0_ge_0",
    )

    # Example STL property:
    # phi := G_[0,2] p
    phi = Always(config.ENCODING_ALWAYS_START, config.ENCODING_ALWAYS_END, safe)

    counters = encode_resilience_counters(
        phi,
        model,
        horizon,
        prefix="safe",
    )

    rec = recoverability_objective(config.ENCODING_ALPHA, counters)
    dur = durability_objective(config.ENCODING_BETA, counters)

    return model, rec, dur


def main():
    if config.RUN_ENCODING_DEMO:
        run_encoding_demo()
        print()
    run_plant_visualization_demo()


def run_encoding_demo():
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


def run_plant_visualization_demo():
    sigma = config.PROCESS_NOISE_VARIANCE
    plant = NoisyLinearPlant.from_parameters(
        a=config.A,
        b=config.B,
        c=config.C,
        initial_state=config.INITIAL_STATE,
        process_noise=config.PROCESS_NOISE,
        process_noise_variance=sigma,
        observation_noise=config.OBSERVATION_NOISE,
        observation_noise_variance=config.OBSERVATION_NOISE_VARIANCE,
        state_lower_bound=config.STATE_LOWER_BOUND,
        state_upper_bound=config.STATE_UPPER_BOUND,
        control_lower_bound=config.CONTROL_LOWER_BOUND,
        control_upper_bound=config.CONTROL_UPPER_BOUND,
        observation_lower_bound=config.OBSERVATION_LOWER_BOUND,
        observation_upper_bound=config.OBSERVATION_UPPER_BOUND,
        seed=config.SEED,
    )
    sensing_policy = AgeOfInformationSensingPolicy(
        a=plant.config.a,
        sigma=sigma,
        noise_threshold=config.SENSING_NOISE_THRESHOLD,
    )
    stl_config = STLResilienceConfig(
        delta=config.STL_DELTA,
        alpha=config.STL_ALPHA,
        beta=config.STL_BETA,
        epsilon=config.STL_EPSILON,
        process_noise_variance=sigma,
        big_m=config.STL_BIG_M,
        norm_type=config.STL_NORM_TYPE,
        solver=config.STL_SOLVER,
    )
    sensing_schedule = []
    estimated_state_history = []
    state_deviation_costs = []
    control_costs = []
    tx_power_costs = []
    last_plan = None
    q = config.STATE_REPORTING_MATRIX

    print("Noisy linear plant demo")
    print("Initial state:", plant.get_state())
    print("Sensing age limit:", sensing_policy.age_limit)

    for time_index in range(config.SIMULATION_STEPS):
        if time_index in config.NOISE_CHANGES:
            change = config.NOISE_CHANGES[time_index]
            if "process_noise" in change or "process_noise_variance" in change:
                sigma = change.get("process_noise_variance", sigma)
                plant.set_process_noise(
                    change.get("process_noise", config.PROCESS_NOISE),
                    variance=sigma,
                )
                sensing_policy.set_sigma(sigma)
            stl_config = STLResilienceConfig(
                delta=stl_config.delta,
                alpha=stl_config.alpha,
                beta=stl_config.beta,
                epsilon=stl_config.epsilon,
                process_noise_variance=sigma,
                big_m=stl_config.big_m,
                norm_type=stl_config.norm_type,
                solver=stl_config.solver,
            )
            if "observation_noise" in change or "observation_noise_variance" in change:
                plant.set_observation_noise(
                    change.get("observation_noise", config.OBSERVATION_NOISE),
                    variance=change.get(
                        "observation_noise_variance",
                        config.OBSERVATION_NOISE_VARIANCE,
                    ),
                )
            print(f"  Changed noise model at step {time_index}")

        sensing_decision = sensing_policy.decide()
        observed_state = plant.get_state() if sensing_decision.sense else None
        control_decision = solve_control_decision(
            observed_state=observed_state,
            sensing_decision=sensing_decision,
            target_state=config.TARGET_STATE,
            control_coefficient_matrix=config.CONTROL_COEFFICIENT_MATRIX,
            tx_power=config.TX_POWER,
            stl_config=stl_config,
            a=plant.config.a,
            b=plant.config.b,
            control_lower_bound=plant.config.control_lower_bound,
            control_upper_bound=plant.config.control_upper_bound,
            last_plan=last_plan,
        )
        last_plan = control_decision.plan
        sensing_schedule.append(sensing_decision.sense)
        step = plant.apply_control(
            control_decision.control,
            sense_after=sensing_decision.sense,
        )
        estimated_next_state = (
            plant.config.a @ control_decision.estimated_state
            + plant.config.b @ control_decision.control
        )
        estimated_state_history.append(estimated_next_state)
        state_error = step.next_state - config.TARGET_STATE
        state_deviation_costs.append(float(state_error.T @ q @ state_error))
        control_costs.append(
            float(step.control.T @ config.CONTROL_COEFFICIENT_MATRIX @ step.control)
        )
        tx_power_costs.append(float(config.TX_POWER) if sensing_decision.sense else 0.0)

        print(f"  Step {time_index}")
        print("    Target state:", config.TARGET_STATE)
        print("    Estimated state:", control_decision.estimated_state)
        print("    Applied control:", step.control)
        print("    Age of information:", sensing_decision.age_of_information)
        print("    Age limit:", sensing_decision.age_limit)
        print("    Sensed:", sensing_decision.sense)
        print("    New plan:", control_decision.planned_from_observation)
        if control_decision.planned_from_observation:
            print("    Planned states:", control_decision.estimated_states)
            print("    Planned controls:", control_decision.controls)
            print("    STL satisfaction:", control_decision.plan.predicate_satisfaction)
            print("    Recovery time:", control_decision.plan.recovery_time)
            print("    Durability time:", control_decision.plan.durability_time)
        print("    Next state:", step.next_state)

    output_path = plot_plant_history(
        plant,
        sensing_schedule=sensing_schedule,
        target_state=config.TARGET_STATE,
        estimated_states=estimated_state_history,
        deviation_limit=config.STL_DELTA,
        deviation_norm=config.STL_NORM_TYPE,
        output_path=config.OUTPUT_PLOT_PATH,
        show=config.SHOW_PLOT,
    )
    print("Noise model changes:")
    for change in plant.noise_change_history:
        print(
            f"  Step {change.time_index}: {change.channel} noise "
            f"set to {change.noise_type} with variance {change.variance}"
        )
    print("Average realized costs:")
    print(f"  State deviation: {np.mean(state_deviation_costs):.6f}")
    print(f"  Control effort:  {np.mean(control_costs):.6f}")
    print(f"  Tx power:        {np.mean(tx_power_costs):.6f}")
    print(f"Saved dynamics plot to: {output_path}")


if __name__ == "__main__":
    main()
