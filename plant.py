"""Customizable noisy linear plant simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np


NoiseSampler = Callable[[np.random.Generator, int], np.ndarray]
NoiseType = Literal["zero", "gaussian"] | NoiseSampler


def zero_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    return np.zeros(size)


def gaussian_noise(variance) -> NoiseSampler:
    """Create zero-mean Gaussian noise from a variance or covariance."""

    variance = np.asarray(variance, dtype=float)

    def sample(rng: np.random.Generator, size: int) -> np.ndarray:
        if variance.shape == ():
            return rng.normal(0.0, np.sqrt(float(variance)), size=size)
        covariance = np.diag(variance) if variance.ndim == 1 else variance
        if covariance.shape != (size, size):
            raise ValueError("Gaussian noise covariance shape does not match signal size.")
        return rng.multivariate_normal(np.zeros(size), covariance)

    return sample


@dataclass(frozen=True)
class PlantConfig:
    """Linear plant parameters.

    Dynamics:
        x[k + 1] = A x[k] + B u[k] + w[k]

    Measurement:
        y[k] = C x[k] + v[k]
    """

    a: np.ndarray
    b: np.ndarray
    initial_state: np.ndarray
    c: np.ndarray | None = None
    process_noise_variance: float | np.ndarray = 0.0
    process_noise: NoiseType = "zero"
    observation_noise_variance: float | np.ndarray = 0.0
    observation_noise: NoiseType = "zero"
    state_lower_bound: np.ndarray | None = None
    state_upper_bound: np.ndarray | None = None
    control_lower_bound: np.ndarray | None = None
    control_upper_bound: np.ndarray | None = None
    observation_lower_bound: np.ndarray | None = None
    observation_upper_bound: np.ndarray | None = None


@dataclass(frozen=True)
class PlantStep:
    time_index: int
    previous_state: np.ndarray
    control: np.ndarray
    process_noise: np.ndarray
    next_state: np.ndarray
    measurement: np.ndarray | None


@dataclass(frozen=True)
class NoiseChange:
    time_index: int
    channel: str
    noise_type: str
    variance: float | np.ndarray


class NoisyLinearPlant:
    """A small simulation environment for noisy linear control dynamics."""

    def __init__(self, config: PlantConfig, *, seed: int | None = None):
        self.config = self._validate_config(config)
        self.rng = np.random.default_rng(seed)
        self.reset()

    @classmethod
    def from_parameters(
        cls,
        *,
        a,
        b,
        initial_state,
        c=None,
        process_noise: NoiseType = "zero",
        process_noise_variance=0.0,
        observation_noise: NoiseType = "zero",
        observation_noise_variance=0.0,
        state_lower_bound=None,
        state_upper_bound=None,
        control_lower_bound=None,
        control_upper_bound=None,
        observation_lower_bound=None,
        observation_upper_bound=None,
        seed: int | None = None,
    ) -> "NoisyLinearPlant":
        """Create a plant directly from runtime parameter values."""

        return cls(
            PlantConfig(
                a=np.asarray(a, dtype=float),
                b=np.asarray(b, dtype=float),
                initial_state=np.asarray(initial_state, dtype=float),
                c=None if c is None else np.asarray(c, dtype=float),
                process_noise=process_noise,
                process_noise_variance=process_noise_variance,
                observation_noise=observation_noise,
                observation_noise_variance=observation_noise_variance,
                state_lower_bound=state_lower_bound,
                state_upper_bound=state_upper_bound,
                control_lower_bound=control_lower_bound,
                control_upper_bound=control_upper_bound,
                observation_lower_bound=observation_lower_bound,
                observation_upper_bound=observation_upper_bound,
            ),
            seed=seed,
        )

    @property
    def state_dim(self) -> int:
        return self.config.a.shape[0]

    @property
    def control_dim(self) -> int:
        return self.config.b.shape[1]

    @property
    def measurement_dim(self) -> int:
        return self.config.c.shape[0]

    @property
    def time_index(self) -> int:
        return len(self.control_history)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        initial_state = self.config.initial_state if state is None else state
        self.state = np.asarray(initial_state, dtype=float).reshape(self.state_dim)
        self.state = self._clip_state(self.state)
        self.state_history = [self.state.copy()]
        self.control_history: list[np.ndarray] = []
        self.measurement_history: list[np.ndarray | None] = []
        self.noise_history: list[np.ndarray] = []
        self.noise_change_history: list[NoiseChange] = []
        return self.get_state()

    def get_state(self, *, copy: bool = True) -> np.ndarray:
        return self.state.copy() if copy else self.state

    def set_process_noise(
        self,
        noise: NoiseType,
        *,
        variance=0.0,
    ) -> None:
        """Change the disturbance model for subsequent plant transitions."""

        sampler = build_noise_sampler(noise, variance, "process_noise")
        self.config = replace(
            self.config,
            process_noise=sampler,
            process_noise_variance=copy_variance(variance),
        )
        self.noise_change_history.append(
            NoiseChange(
                time_index=self.time_index,
                channel="process",
                noise_type=noise_label(noise),
                variance=copy_variance(variance),
            )
        )

    def set_observation_noise(
        self,
        noise: NoiseType,
        *,
        variance=0.0,
    ) -> None:
        """Change the sensor-noise model for subsequent measurements."""

        sampler = build_noise_sampler(noise, variance, "observation_noise")
        self.config = replace(
            self.config,
            observation_noise=sampler,
            observation_noise_variance=copy_variance(variance),
        )
        self.noise_change_history.append(
            NoiseChange(
                time_index=self.time_index,
                channel="observation",
                noise_type=noise_label(noise),
                variance=copy_variance(variance),
            )
        )

    def sense(self) -> np.ndarray:
        measurement_noise = self.config.observation_noise(self.rng, self.measurement_dim)
        measurement = self.config.c @ self.state + measurement_noise
        measurement = clip_vector(
            measurement,
            self.config.observation_lower_bound,
            self.config.observation_upper_bound,
        )
        self.measurement_history.append(measurement.copy())
        return measurement

    def apply_control(self, control, *, sense_after: bool = False) -> PlantStep:
        control = np.asarray(control, dtype=float).reshape(self.control_dim)
        control = self._clip_control(control)

        time_index = self.time_index
        previous_state = self.state.copy()
        process_noise = self.config.process_noise(self.rng, self.state_dim)
        next_state = self.config.a @ previous_state + self.config.b @ control + process_noise
        next_state = self._clip_state(next_state)

        self.state = next_state
        self.control_history.append(control.copy())
        self.noise_history.append(process_noise.copy())
        self.state_history.append(next_state.copy())

        measurement = self.sense() if sense_after else None
        if measurement is None:
            self.measurement_history.append(None)

        return PlantStep(
            time_index=time_index,
            previous_state=previous_state,
            control=control,
            process_noise=process_noise,
            next_state=next_state.copy(),
            measurement=measurement,
        )

    def rollout(self, controls, *, sensing_schedule=None) -> list[PlantStep]:
        controls = list(controls)
        if sensing_schedule is None:
            sensing_schedule = [False] * len(controls)
        sensing_schedule = list(sensing_schedule)
        if len(controls) != len(sensing_schedule):
            raise ValueError("controls and sensing_schedule must have the same length.")

        steps = []
        for control, should_sense in zip(controls, sensing_schedule):
            steps.append(self.apply_control(control, sense_after=bool(should_sense)))
        return steps

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        return clip_vector(
            state,
            self.config.state_lower_bound,
            self.config.state_upper_bound,
        )

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        return clip_vector(
            control,
            self.config.control_lower_bound,
            self.config.control_upper_bound,
        )

    @staticmethod
    def _validate_config(config: PlantConfig) -> PlantConfig:
        a = np.asarray(config.a, dtype=float)
        b = np.asarray(config.b, dtype=float)
        initial_state = np.asarray(config.initial_state, dtype=float).reshape(-1)

        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("A must be a square matrix.")
        if b.ndim != 2 or b.shape[0] != a.shape[0]:
            raise ValueError("B must have one row per state dimension.")
        if initial_state.size != a.shape[0]:
            raise ValueError("initial_state size must match A.")

        c = np.eye(a.shape[0]) if config.c is None else np.asarray(config.c, dtype=float)
        if c.ndim != 2 or c.shape[1] != a.shape[0]:
            raise ValueError("C must have one column per state dimension.")

        state_lower = reshape_optional(config.state_lower_bound, a.shape[0])
        state_upper = reshape_optional(config.state_upper_bound, a.shape[0])
        control_lower = reshape_optional(config.control_lower_bound, b.shape[1])
        control_upper = reshape_optional(config.control_upper_bound, b.shape[1])
        observation_lower = reshape_optional(config.observation_lower_bound, c.shape[0])
        observation_upper = reshape_optional(config.observation_upper_bound, c.shape[0])

        validate_bounds(state_lower, state_upper, "state")
        validate_bounds(control_lower, control_upper, "control")
        validate_bounds(observation_lower, observation_upper, "observation")

        return PlantConfig(
            a=a,
            b=b,
            initial_state=initial_state,
            c=c,
            process_noise_variance=config.process_noise_variance,
            process_noise=build_noise_sampler(
                config.process_noise,
                config.process_noise_variance,
                "process_noise",
            ),
            observation_noise_variance=config.observation_noise_variance,
            observation_noise=build_noise_sampler(
                config.observation_noise,
                config.observation_noise_variance,
                "observation_noise",
            ),
            state_lower_bound=state_lower,
            state_upper_bound=state_upper,
            control_lower_bound=control_lower,
            control_upper_bound=control_upper,
            observation_lower_bound=observation_lower,
            observation_upper_bound=observation_upper,
        )


def build_noise_sampler(noise: NoiseType, variance, name: str) -> NoiseSampler:
    if callable(noise):
        return noise
    if noise == "zero":
        return zero_noise
    if noise == "gaussian":
        variance = np.asarray(variance, dtype=float)
        validate_variance(variance, name)
        return gaussian_noise(variance)
    raise ValueError(f"{name} must be 'zero', 'gaussian', or a callable sampler.")


def copy_variance(variance) -> float | np.ndarray:
    variance = np.asarray(variance, dtype=float)
    if variance.shape == ():
        return float(variance)
    return variance.copy()


def noise_label(noise: NoiseType) -> str:
    return noise if isinstance(noise, str) else "callable"


def validate_variance(variance: np.ndarray, name: str) -> None:
    if variance.ndim <= 1:
        if np.any(variance < 0):
            raise ValueError(f"{name}_variance must be non-negative.")
        return
    if variance.ndim != 2 or variance.shape[0] != variance.shape[1]:
        raise ValueError(f"{name}_variance must be a scalar, vector, or square matrix.")
    if not np.allclose(variance, variance.T):
        raise ValueError(f"{name}_variance covariance matrix must be symmetric.")
    if np.min(np.linalg.eigvalsh(variance)) < -1e-10:
        raise ValueError(f"{name}_variance covariance matrix must be positive semidefinite.")


def reshape_optional(value, size: int) -> np.ndarray | None:
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape == ():
        return np.full(size, float(value))
    return value.reshape(size)


def validate_bounds(
    lower_bound: np.ndarray | None,
    upper_bound: np.ndarray | None,
    name: str,
) -> None:
    if lower_bound is not None and upper_bound is not None:
        if np.any(lower_bound > upper_bound):
            raise ValueError(f"{name}_lower_bound cannot exceed {name}_upper_bound.")


def clip_vector(
    value: np.ndarray,
    lower_bound: np.ndarray | None,
    upper_bound: np.ndarray | None,
) -> np.ndarray:
    clipped = value.copy()
    if lower_bound is not None:
        clipped = np.maximum(clipped, lower_bound)
    if upper_bound is not None:
        clipped = np.minimum(clipped, upper_bound)
    return clipped
