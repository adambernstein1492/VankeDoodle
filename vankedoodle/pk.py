"""Linear compartmental PK solver (one- or two-compartment, IV).

Concentrations and exposure are obtained from the exact solution of the linear
state-space system, augmented with a final state holding the running integral of
the central-compartment concentration.  This gives analytically exact AUC over
any interval, handles arbitrary (irregular, overlapping, missed) dosing
histories, and never requires a steady-state assumption.

The same machinery serves both structural models: a one-compartment model is
simply a smaller state matrix, so nothing downstream -- estimation, exposure,
dose selection -- needs to know how many compartments a population model has.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .core import Dose

_ROUND = 9  # hours, for caching propagators of identical segment durations


class CompartmentalModel:
    """Exact solver for a linear IV compartmental model.

    Construct through :meth:`one_compartment` / :meth:`two_compartment`, or pass
    a parameter dict to :func:`model_from_params`.
    """

    def __init__(self, A: np.ndarray, V1: float, parameters: Dict[str, float]):
        self._A = np.asarray(A, dtype=float)
        self.V1 = float(V1)
        self.parameters = dict(parameters)
        self.n_states = self._A.shape[0]            # includes the AUC state
        self.n_compartments = self.n_states - 1
        self._cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    # -- constructors ---------------------------------------------------
    @staticmethod
    def _check(**pars) -> None:
        for nm, val in pars.items():
            if val is None or not np.isfinite(val) or val <= 0:
                raise ValueError(f"{nm} must be a finite positive number (got {val!r}).")

    @classmethod
    def one_compartment(cls, CL: float, V: float) -> "CompartmentalModel":
        cls._check(CL=CL, V=V)
        k10 = CL / V
        A = np.array([[-k10, 0.0],
                      [1.0 / V, 0.0]], dtype=float)
        return cls(A, V, {"CL": float(CL), "V": float(V)})

    @classmethod
    def two_compartment(cls, CL: float, V1: float, Q: float, V2: float) -> "CompartmentalModel":
        cls._check(CL=CL, V1=V1, Q=Q, V2=V2)
        k10, k12, k21 = CL / V1, Q / V1, Q / V2
        A = np.array([
            [-(k10 + k12), k21, 0.0],
            [k12,          -k21, 0.0],
            [1.0 / V1,      0.0, 0.0],
        ], dtype=float)
        return cls(A, V1, {"CL": float(CL), "V1": float(V1),
                           "Q": float(Q), "V2": float(V2)})

    # -- propagators ----------------------------------------------------
    def _propagator(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (E, J) with E = exp(A*dt) and J = int_0^dt exp(A*s) ds."""
        key = round(float(dt), _ROUND)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = self.n_states
        block = np.zeros((2 * n, 2 * n))
        block[:n, :n] = self._A
        block[:n, n:] = np.eye(n)
        ex = expm(block * key)
        E, J = ex[:n, :n], ex[:n, n:]
        self._cache[key] = (E, J)
        return E, J

    # -- simulation -----------------------------------------------------
    def simulate(self, doses: Sequence[Dose], times: Sequence[float],
                 initial_state: Sequence[float] | None = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Concentrations (mg/L) and cumulative AUC (mg*h/L) at ``times``.

        Cumulative AUC is measured from the start of the simulation, which is the
        earliest of the first dose and the first requested time.
        """
        times = np.asarray(times, dtype=float)
        doses = sorted(doses, key=lambda d: d.start_h)
        if times.size == 0:
            return np.empty(0), np.empty(0)
        n = self.n_states

        events = [d.start_h for d in doses] + [d.end_h for d in doses]
        t_start = min([*events, float(times.min())]) if events else float(times.min())
        grid = sorted({t_start, *events, *times.tolist()})
        grid = [t for t in grid if t >= t_start - 1e-12]

        state = np.zeros(n) if initial_state is None else np.asarray(initial_state, float).copy()
        conc = np.full(times.shape, np.nan)
        cum = np.full(times.shape, np.nan)
        want: Dict[float, list[int]] = {}
        for i, t in enumerate(times):
            want.setdefault(round(float(t), _ROUND), []).append(i)

        def record(t: float):
            for i in want.get(round(float(t), _ROUND), []):
                conc[i] = state[0] / self.V1
                cum[i] = state[-1]

        for d in doses:
            if d.duration_h == 0 and abs(d.start_h - grid[0]) < 1e-12:
                state[0] += d.amount_mg
        record(grid[0])

        inflow = np.zeros(n)
        for left, right in zip(grid[:-1], grid[1:]):
            dt = right - left
            if dt <= 0:
                continue
            rate = 0.0
            mid = 0.5 * (left + right)
            for d in doses:
                if d.duration_h > 0 and d.start_h <= mid < d.end_h:
                    rate += d.rate_mg_per_h
            E, J = self._propagator(dt)
            state = E @ state
            if rate:
                inflow[:] = 0.0
                inflow[0] = rate
                state = state + J @ inflow
            for d in doses:
                if d.duration_h == 0 and abs(d.start_h - right) < 1e-12:
                    state[0] += d.amount_mg
            record(right)

        return conc, cum

    # -- convenience ----------------------------------------------------
    def concentrations(self, doses: Sequence[Dose], times: Sequence[float]) -> np.ndarray:
        return self.simulate(doses, times)[0]

    def auc(self, doses: Sequence[Dose], t_start: float, t_end: float) -> float:
        """Exact AUC (mg*h/L) between two times for the given dosing history."""
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start.")
        _, cum = self.simulate(doses, [t_start, t_end])
        return float(cum[1] - cum[0])

    def auc24_steady_state(self, total_daily_dose_mg: float) -> float:
        """Steady-state AUC over 24 h = total daily dose / clearance."""
        return float(total_daily_dose_mg) / self.CL

    @property
    def CL(self) -> float:
        return self.parameters["CL"]

    @property
    def volume_central(self) -> float:
        return self.V1

    @property
    def volume_steady_state(self) -> float:
        return self.V1 + self.parameters.get("V2", 0.0)

    @property
    def rate_constants(self) -> np.ndarray:
        """Disposition rate constants (1/h), fastest first."""
        drug = self._A[:self.n_compartments, :self.n_compartments]
        return np.sort(-np.linalg.eigvals(drug).real)[::-1]

    @property
    def terminal_half_life(self) -> float:
        """Half-life of the slowest disposition phase (h)."""
        return float(np.log(2) / self.rate_constants[-1])

    @property
    def half_lives(self) -> Tuple[float, ...]:
        """Half-life of each disposition phase, fastest first."""
        return tuple(float(np.log(2) / k) for k in self.rate_constants)


class TwoCompartmentModel(CompartmentalModel):
    """Backwards-compatible constructor: ``TwoCompartmentModel(CL, V1, Q, V2)``."""

    def __init__(self, CL: float, V1: float, Q: float, V2: float):
        base = CompartmentalModel.two_compartment(CL, V1, Q, V2)
        super().__init__(base._A, base.V1, base.parameters)


class OneCompartmentModel(CompartmentalModel):
    """``OneCompartmentModel(CL, V)``."""

    def __init__(self, CL: float, V: float):
        base = CompartmentalModel.one_compartment(CL, V)
        super().__init__(base._A, base.V1, base.parameters)


def model_from_params(pars: Dict[str, float]) -> CompartmentalModel:
    """Build the solver implied by the parameter names present.

    ``{CL, V}`` gives a one-compartment model; ``{CL, V1, Q, V2}`` a
    two-compartment model.
    """
    keys = set(pars)
    if {"CL", "V1", "Q", "V2"} <= keys:
        return CompartmentalModel.two_compartment(pars["CL"], pars["V1"],
                                                  pars["Q"], pars["V2"])
    if {"CL", "V"} <= keys:
        return CompartmentalModel.one_compartment(pars["CL"], pars["V"])
    if {"CL", "V1"} <= keys:
        return CompartmentalModel.one_compartment(pars["CL"], pars["V1"])
    raise ValueError(
        "Cannot build a structural model from parameters "
        f"{sorted(keys)}; expected {{CL, V}} or {{CL, V1, Q, V2}}."
    )


def time_grid(t0: float, t1: float, doses, n_points: int = 600) -> np.ndarray:
    """Evenly spaced times from ``t0`` to ``t1`` plus every infusion start and end
    in between, so that each peak and trough falls exactly on the grid and a
    plotted curve never cuts a peak short."""
    base = np.linspace(t0, t1, n_points)
    events = [t for d in doses for t in (d.start_h, d.end_h) if t0 <= t <= t1]
    return np.unique(np.concatenate([base, np.asarray(events, dtype=float)]))
