"""Editable parameters for the STL resilient-control simulation.

Change values in this file to run a new experiment. The remaining modules
provide plant, controller, and visualization behavior.
"""

import numpy as np


# Entry-point behavior.
RUN_ENCODING_DEMO = False
SIMULATION_STEPS = 120  # 250
OUTPUT_PLOT_PATH = "main_plant_dynamics.png"
SHOW_PLOT = False


# Plant dynamics: x[k + 1] = A x[k] + B u[k] + w[k].
A = np.array([[1.1]], dtype=float)
B = np.array([[-0.25]], dtype=float)
C = None
INITIAL_STATE = np.array([3.0], dtype=float)  # 0
SEED = 17


# Plant noise models and bounds.
PROCESS_NOISE = "gaussian"
PROCESS_NOISE_VARIANCE = 0.01  # 0.005
OBSERVATION_NOISE = "zero"
OBSERVATION_NOISE_VARIANCE = 0.05

STATE_LOWER_BOUND = None  # -10.0
STATE_UPPER_BOUND = None  # 10.0
CONTROL_LOWER_BOUND = -5.0
CONTROL_UPPER_BOUND = 5.0
OBSERVATION_LOWER_BOUND = None  # -10.0
OBSERVATION_UPPER_BOUND = None  # 10.0


# Changes introduced while the simulation is running.
# Each entry may contain process_noise, process_noise_variance,
# observation_noise, and observation_noise_variance.
NOISE_CHANGES = {
    41: {  # 151
        "process_noise": "gaussian",
        "process_noise_variance": 0.05, # 0.025
        "observation_noise": "zero",
        "observation_noise_variance": 0.05,
    },
}


# Age-of-information sensing policy.
SENSING_NOISE_THRESHOLD = .5  # 0.1


# Tracking target and cost-reporting matrices.
TARGET_STATE = np.array([0.0], dtype=float)  # 2
STATE_REPORTING_MATRIX = np.array([[0.0]], dtype=float)  # 1
CONTROL_COEFFICIENT_MATRIX = np.array([[0.2]], dtype=float)
TX_POWER = 0.15
PLANNING_HORIZON = 25  # 25


# Resilient STL chance constraint:
# R_(ALPHA, BETA)(||estimated_state - TARGET_STATE|| <= DELTA).
STL_DELTA = 1  # .5
STL_ALPHA = 6  # 4
STL_BETA = 5  # 8
STL_EPSILON = 0.01  # .2
STL_ROBUST_IGNORE = False
STL_BIG_M = 1_000.0
STL_NORM_TYPE = "infinity"
STL_SOLVER = "SCIP"


# Optional symbolic STL/MILP diagnostic settings.
ENCODING_HORIZON = 5
ENCODING_ALPHA = 2
ENCODING_BETA = 2
ENCODING_BIG_M = 1_000
ENCODING_STATE_LOWER_BOUND = -10.0
ENCODING_STATE_UPPER_BOUND = 10.0
ENCODING_ALWAYS_START = 0
ENCODING_ALWAYS_END = 2
