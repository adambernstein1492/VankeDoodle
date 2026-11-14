"""Maximum a posteriori (MAP) Bayesian estimation of individual PK parameters.

This is the estimation method used by model-informed precision dosing tools:
the population model supplies the prior, the patient's measured concentrations
supply the likelihood, and the individual random effects (etas) are the
posterior mode.  With no levels the result is the population prediction; each
level pulls the estimate toward the individual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_t

from .core import Dose, Level, Patient
from .models import PopPKModel, get_model
from .pk import CompartmentalModel, model_from_params

_LOG2PI = float(np.log(2.0 * np.pi))

# Posterior sampling. The Laplace (normal) approximation at the MAP is symmetric
# in the etas, but with few levels the true posterior is not: clearance in
# particular has a long tail, so normal draws place the AUC interval too low.
# Draws are therefore taken by adaptive importance sampling: a heavy-tailed
# proposal around the MAP gives a first estimate of the posterior's mean and
# covariance, each later stage's proposal is built from the previous estimate,
# and the last stage's draws, weighted by the exact posterior over the proposal
# density, form a pool that is resampled into equally weighted draws. Three
# stages keep the effective sample size above ~2000 in the hardest cases tested.
SAMPLING_STAGES = ((2000, 1.3), (2000, 1.3), (6000, 1.3))   # (draws, spread) per stage
PROPOSAL_DF = 5                 # multivariate t: heavier tails than the posterior
MIN_EFFECTIVE_DRAWS = 500       # below this the weights are unreliable
POOL_SEED = 20240611            # the pool is fixed for a fit; ``seed`` picks from it


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
    _objective: Optional[Callable] = field(default=None, repr=False, compare=False)
    _pool: Optional[tuple] = field(default=None, repr=False, compare=False)
    _pool_built: bool = field(default=False, repr=False, compare=False)

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

    # -- posterior draws -----------------------------------------------
    def _posterior_pool(self) -> Optional[tuple]:
        """Importance-weighted draws ``(etas, weights, ess)``, built once per fit.

        None when the normal draws are exact (no levels: the posterior is the
        population prior) or the weights proved unreliable (a warning is added).
        """
        if self._pool_built:
            return self._pool
        self._pool_built = True
        if not self.levels or self._objective is None or self.eta_covariance is None:
            return None
        rng = np.random.default_rng(POOL_SEED)
        # the first stage is centred on the MAP with the Laplace covariance
        loc, shape = self.eta, self.eta_covariance * SAMPLING_STAGES[0][1] ** 2
        for i, (n, _) in enumerate(SAMPLING_STAGES):
            draws, w, ess = self._weighted_draws(loc, shape, n, rng)
            last = i == len(SAMPLING_STAGES) - 1
            if last or ess < 50:            # too few to re-estimate the moments
                continue
            mean = w @ draws
            centred = draws - mean
            cov = (centred * w[:, None]).T @ centred
            if np.all(np.linalg.eigvalsh(cov) > 0):
                # a t's covariance is shape * df / (df - 2)
                spread = SAMPLING_STAGES[i + 1][1]
                loc, shape = mean, cov * (PROPOSAL_DF - 2) / PROPOSAL_DF * spread ** 2
        if ess < MIN_EFFECTIVE_DRAWS:
            self.warnings.append(
                "The posterior could not be sampled reliably; uncertainty intervals use "
                "a normal approximation and may be too narrow or shifted.")
            return None
        self._pool = (draws, w, ess)
        return self._pool

    def _weighted_draws(self, loc, shape, n, rng):
        """Draws from a multivariate t proposal, with normalised importance weights
        (exact posterior over proposal density) and their effective sample size."""
        k = len(self.eta)
        proposal = multivariate_t(loc=loc, shape=shape, df=PROPOSAL_DF, seed=rng)
        draws = np.asarray(proposal.rvs(size=n), dtype=float).reshape(-1, k)
        log_w = -0.5 * np.array([self._objective(e) for e in draws]) - proposal.logpdf(draws)
        log_w[~np.isfinite(log_w)] = -np.inf
        if not np.any(np.isfinite(log_w)):
            return draws, np.full(n, 1.0 / n), 0.0
        w = np.exp(log_w - np.max(log_w))
        w /= w.sum()
        return draws, w, float(1.0 / np.sum(w ** 2))

    @property
    def posterior_method(self) -> str:
        """How posterior draws are made: "prior" (no levels; normal and exact),
        "importance sampling", "normal approximation" (the fallback), or "none"."""
        if self.eta_covariance is None:
            return "none"
        if not self.levels:
            return "prior"
        return "importance sampling" if self._posterior_pool() else "normal approximation"

    @property
    def posterior_ess(self) -> Optional[float]:
        """Effective number of independent draws behind the importance weights."""
        pool = self._posterior_pool()
        return pool[2] if pool else None

    @property
    def posterior_covariance(self) -> Optional[np.ndarray]:
        """Covariance of the etas under the posterior actually sampled."""
        pool = self._posterior_pool()
        if pool is None:
            return self.eta_covariance
        draws, w, _ = pool
        mean = w @ draws
        centred = draws - mean
        return (centred * w[:, None]).T @ centred

    def sample_etas(self, n: int = 2000, seed: Optional[int] = 0) -> np.ndarray:
        """Equally weighted draws from the eta posterior.

        Resampled from the importance-weighted pool when it is available;
        otherwise drawn from the normal (Laplace) approximation, which is exact
        when there are no levels.
        """
        if self.eta_covariance is None:
            raise RuntimeError("Posterior covariance unavailable for this fit.")
        rng = np.random.default_rng(seed)
        pool = self._posterior_pool()
        if pool is None:
            return rng.multivariate_normal(self.eta, self.eta_covariance, size=int(n))
        draws, w, _ = pool
        return draws[rng.choice(len(w), size=int(n), replace=True, p=w)]

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
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                val += float(np.sum((obs - pred) ** 2 / var + np.log(2.0 * np.pi * var)))
        # far from the mode the variance can overflow; treat that as impossible
        return val if np.isfinite(val) else 1e12

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
                  message=str(best.message), warnings=warn, _objective=obj)


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
