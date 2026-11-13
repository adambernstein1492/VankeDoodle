"""Maximum a posteriori (MAP) Bayesian estimation of individual PK parameters.

This is the estimation method used by model-informed precision dosing tools:
the population model supplies the prior, the patient's measured concentrations
supply the likelihood, and the individual random effects (etas) are the
posterior mode.  With no levels the result is the population prediction; each
level pulls the estimate toward the individual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .core import Dose, Level, Patient
from .models import PopPKModel, get_model
from .pk import CompartmentalModel, model_from_params

_LOG2PI = float(np.log(2.0 * np.pi))

# A MAP estimate this far from the population value, measured in prior standard
# deviations, means the levels and the dosing history cannot both be right.
# Real between-patient variation rarely exceeds 3 SD; a mistyped dose, wrong
# concentration units or a mis-entered sample time easily produces 10 or more.
IMPLAUSIBLE_Z = 3.0


@dataclass
class MAPFit:
    """Result of a MAP Bayesian fit."""
    patient: Patient
    model: PopPKModel
    doses: list
    levels: list
    eta: np.ndarray
    parameters: Dict[str, float]           # individual CL, V1, Q, V2
    population_parameters: Dict[str, float]
    eta_covariance: Optional[np.ndarray]   # Laplace approximation of posterior cov
    objective: float                       # -2 log posterior at the mode
    converged: bool
    message: str = ""
    warnings: list = field(default_factory=list)
    implausible: bool = False
    """True when the levels cannot be reconciled with the recorded doses.

    The estimate is still returned so the user can see what the data imply,
    but it must not be presented as a finding or used to recommend a dose.
    """

    # -- derived quantities --------------------------------------------
    @property
    def pk(self) -> CompartmentalModel:
        return model_from_params(self.parameters)

    @property
    def cl(self) -> float:
        return self.parameters["CL"]

    @property
    def cl_per_kg(self) -> float:
        return self.parameters["CL"] / self.patient.weight_kg

    def predict(self, times: Sequence[float]) -> np.ndarray:
        return self.pk.concentrations(self.doses, times)

    def predictions_at_levels(self) -> np.ndarray:
        if not self.levels:
            return np.empty(0)
        return self.predict([l.time_h for l in self.levels])

    def residuals(self) -> Dict[str, np.ndarray]:
        """Observed, predicted and weighted residuals at the measured levels."""
        obs = np.array([l.conc_mg_l for l in self.levels], dtype=float)
        pred = self.predictions_at_levels()
        sd = np.sqrt((self.model.sigma_prop * pred) ** 2 + self.model.sigma_add ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            wres = (obs - pred) / sd
            pct = 100.0 * (pred - obs) / obs
        return {"observed": obs, "predicted": pred,
                "weighted_residual": wres, "percent_error": pct}

    def auc(self, t_start: float, t_end: float) -> float:
        return self.pk.auc(self.doses, t_start, t_end)

    def sample_etas(self, n: int = 2000, seed: Optional[int] = 0) -> np.ndarray:
        """Draw from the Laplace (normal) approximation to the eta posterior."""
        if self.eta_covariance is None:
            raise RuntimeError("Posterior covariance unavailable for this fit.")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.eta, self.eta_covariance, size=int(n))

    def parameter_samples(self, n: int = 2000, seed: Optional[int] = 0) -> list:
        return [self.model.individual(self.patient, e)
                for e in self.sample_etas(n=n, seed=seed)]


# --------------------------------------------------------------------------
def _objective_factory(patient, model, doses, levels):
    omega_inv = np.linalg.inv(model.omega)
    _, logdet_omega = np.linalg.slogdet(model.omega)
    n_eta = model.n_eta
    obs = np.array([l.conc_mg_l for l in levels], dtype=float)
    times = [l.time_h for l in levels]

    def obj(eta: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        pars = model.individual(patient, eta)
        try:
            pk = model_from_params(pars)
        except ValueError:
            return 1e12
        val = float(eta @ omega_inv @ eta) + n_eta * _LOG2PI + logdet_omega
        if obs.size:
            pred = pk.concentrations(doses, times)
            if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
                return 1e12
            var = (model.sigma_prop * pred) ** 2 + model.sigma_add ** 2
            val += float(np.sum((obs - pred) ** 2 / var + np.log(2.0 * np.pi * var)))
        return val

    return obj


def _numeric_hessian(fun, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    n = x.size
    h = np.maximum(np.abs(x), 1.0) * step
    H = np.zeros((n, n))
    f0 = fun(x)
    for i in range(n):
        for j in range(i, n):
            xi = np.zeros(n); xi[i] = h[i]
            xj = np.zeros(n); xj[j] = h[j]
            if i == j:
                H[i, i] = (fun(x + xi) - 2.0 * f0 + fun(x - xi)) / (h[i] ** 2)
            else:
                fpp = fun(x + xi + xj); fpm = fun(x + xi - xj)
                fmp = fun(x - xi + xj); fmm = fun(x - xi - xj)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    return H


def fit(patient: Patient,
        doses: Sequence[Dose],
        levels: Sequence[Level] = (),
        model: str | PopPKModel = "smit2021",
        posterior_covariance: bool = True) -> MAPFit:
    """Estimate individual PK parameters by MAP Bayesian estimation.

    Parameters
    ----------
    patient : Patient
    doses : the full dosing history actually administered
    levels : measured concentrations (may be empty -> population prediction)
    model : population model name or object used as the prior
    """
    mdl = get_model(model)
    doses = list(doses)
    levels = sorted(levels, key=lambda l: l.time_h)
    warn = list(mdl.check(patient))

    for lv in levels:
        if doses and lv.time_h < min(d.start_h for d in doses):
            raise ValueError(
                f"Level at t={lv.time_h:g} h precedes the first dose; check the times."
            )

    obj = _objective_factory(patient, mdl, doses, levels)
    n_eta = mdl.n_eta

    if not levels:
        eta = np.zeros(n_eta)
        pars = mdl.individual(patient, eta)
        warn.append("No levels supplied: parameters are the population (prior) values.")
        return MAPFit(patient=patient, model=mdl, doses=doses, levels=list(levels),
                      eta=eta, parameters=pars,
                      population_parameters=mdl.typical(patient),
                      eta_covariance=mdl.omega.copy(), objective=float(obj(eta)),
                      converged=True, message="prior only", warnings=warn)

    best = None
    starts = [np.zeros(n_eta)]
    scale = np.sqrt(np.diag(mdl.omega))
    for sgn in (-1.0, 1.0):
        starts.append(sgn * 0.5 * scale)
    for x0 in starts:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        res = minimize(obj, res.x, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    eta = np.asarray(best.x, dtype=float)
    pars = mdl.individual(patient, eta)

    cov = None
    if posterior_covariance:
        try:
            H = _numeric_hessian(obj, eta)          # Hessian of -2 log posterior
            cov = 2.0 * np.linalg.inv(H)            # => posterior covariance
            vals = np.linalg.eigvalsh(cov)
            if np.any(vals <= 0):
                cov = None
        except np.linalg.LinAlgError:
            cov = None
        if cov is None:
            warn.append("Posterior covariance could not be computed; "
                        "uncertainty intervals are unavailable.")

    # Is this estimate even physiologically possible under the prior?
    sd_prior = np.sqrt(np.diag(mdl.omega))
    z = eta / sd_prior
    worst = int(np.argmax(np.abs(z)))
    implausible = bool(len(levels) and abs(z[worst]) > IMPLAUSIBLE_Z)
    if implausible:
        name = mdl.eta_params[worst]
        fold = float(np.exp(abs(eta[worst])))
        direction = "below" if eta[worst] < 0 else "above"
        warn.append(
            f"The measured levels cannot be reconciled with the doses recorded: "
            f"fitting them puts {name} {fold:,.0f}-fold {direction} the population "
            f"value ({abs(z[worst]):.1f} standard deviations away, where real "
            f"patients rarely exceed 3). Check the dose amounts, the concentration "
            f"units and the sample times before using this result."
        )

    shrink = None
    if cov is not None:
        shrink = 1.0 - np.sqrt(np.diag(cov) / np.diag(mdl.omega))
        if shrink[0] > 0.9:
            warn.append("The levels carry little information about clearance "
                        f"(shrinkage {shrink[0]*100:.0f}%); the estimate is driven by the prior.")

    return MAPFit(patient=patient, model=mdl, doses=doses, levels=list(levels),
                  eta=eta, parameters=pars,
                  population_parameters=mdl.typical(patient),
                  eta_covariance=cov, objective=float(best.fun),
                  converged=bool(best.success or np.isfinite(best.fun)),
                  message=str(best.message), warnings=warn,
                  implausible=implausible)


def simulate_levels(patient: Patient, doses: Sequence[Dose], times: Sequence[float],
                    model: str | PopPKModel = "smit2021",
                    eta: Optional[Sequence[float]] = None,
                    residual_error: bool = True,
                    seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Simulate concentrations for a virtual subject.

    Intended for software testing and for teaching; it is not a substitute for
    measured patient data.
    """
    mdl = get_model(model)
    rng = np.random.default_rng(seed)
    if eta is None:
        eta = rng.multivariate_normal(np.zeros(mdl.n_eta), mdl.omega)
    pars = mdl.individual(patient, eta)
    pk = model_from_params(pars)
    true_conc = pk.concentrations(doses, times)
    obs = true_conc.copy()
    if residual_error:
        # the same combined error model the likelihood assumes: proportional on the
        # prediction plus, where the model has one, an additive component
        obs = true_conc * (1.0 + rng.normal(0.0, mdl.sigma_prop, size=true_conc.shape))
        if mdl.sigma_add:
            obs = obs + rng.normal(0.0, mdl.sigma_add, size=true_conc.shape)
        obs = np.maximum(obs, 1e-3)
    return {"times": np.asarray(times, float), "true": true_conc,
            "observed": obs, "eta": np.asarray(eta, float), "parameters": pars}
