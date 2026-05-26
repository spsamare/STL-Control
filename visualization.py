"""Visualization helpers for noisy linear plant simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_plant_history(
    plant,
    *,
    sensing_schedule=None,
    target_controls=None,
    target_state=None,
    estimated_states=None,
    deviation_limit=None,
    deviation_norm="infinity",
    output_path="plant_dynamics.png",
    show=False,
):
    """Plot target-deviation norms, controls, and sensing decisions."""

    states = np.asarray(plant.state_history)
    controls = np.asarray(plant.control_history)
    measurements = plant.measurement_history
    steps = np.arange(len(controls))
    state_times = np.arange(len(states))

    if sensing_schedule is None:
        sensing_schedule = [measurement is not None for measurement in measurements]
    sensing_schedule = np.asarray(sensing_schedule, dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    if target_state is None:
        raise ValueError("target_state is required to plot state deviation norms.")
    target_state = np.asarray(target_state, dtype=float).reshape(states.shape[1])
    true_deviation = row_norm(states - target_state, deviation_norm)
    axes[0].plot(
        state_times,
        true_deviation,
        marker="o",
        label="true deviation",
    )

    if estimated_states is not None:
        estimated_states = np.asarray(estimated_states, dtype=float)
        if estimated_states.ndim != 2 or estimated_states.shape[1] != states.shape[1]:
            raise ValueError("estimated_states must have one column per state.")
        estimated_times = np.arange(1, len(estimated_states) + 1)
        estimated_deviation = row_norm(estimated_states - target_state, deviation_norm)
        axes[0].plot(
            estimated_times,
            estimated_deviation,
            linestyle=":",
            marker="s",
            label="estimated deviation",
        )

    if deviation_limit is not None:
        axes[0].axhline(
            float(deviation_limit),
            linestyle="--",
            color="tab:red",
            label="STL delta",
        )

    axes[0].set_title("Deviation from target state")
    axes[0].set_ylabel(f"Error norm ({deviation_norm})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    if controls.size:
        for control_index in range(controls.shape[1]):
            axes[1].step(
                steps,
                controls[:, control_index],
                where="post",
                label=f"u[{control_index}]",
            )

    if target_controls is not None:
        axes[1].plot(
            steps,
            target_controls,
            linestyle="--",
            color="black",
            label="target u",
        )

    axes[1].set_title("Control decisions")
    axes[1].set_ylabel("Control value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].step(
        steps,
        sensing_schedule.astype(int),
        where="post",
        color="tab:green",
        label="sense",
    )
    axes[2].set_title("Sensing decisions")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Sense")
    axes[2].set_yticks([0, 1])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def row_norm(values, norm_type):
    if norm_type == "infinity":
        return np.linalg.norm(values, ord=np.inf, axis=1)
    if norm_type == "l2":
        return np.linalg.norm(values, ord=2, axis=1)
    raise ValueError("deviation_norm must be 'infinity' or 'l2'.")
