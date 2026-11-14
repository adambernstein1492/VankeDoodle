"""Predictive summaries for display: bands, distributions and forecast curves.

Everything here draws individual parameter sets from the normal approximation
held on a :class:`~vankedoodle.bayes.MAPFit` -- the Laplace posterior after a fit,
or the population variability itself when the fit has no levels -- and pushes
each draw through the exact solver. Nothing here changes an estimate; it only
describes the uncertainty around one.

Draws are seeded, so the same case always gives the same bands and
probabilities (a page that re-runs on every click should not flicker).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bayes import MAPFit
from .core import Dose
from .dosing import EVALUATION_DAY, Exposure, _regimen_doses
from .pk import model_from_params, time_grid

BAND_SAMPLES = 200          # parameter draws behind a concentration band
AUC_SAMPLES = 4000          # draws behind an AUC24 distribution (clearance only: cheap)
PEAK_SAMPLES = 300          # draws behind a peak distribution (one solve each)
BAND_LEVEL = 0.95           # central coverage of the concentration bands
HISTOGRAM_BIN_WIDTH = 10.0  # mg*h/L, so bin edges fall on the usual 400 and 600


def cv_percent(variances: Sequence[float]) -> List[float]:
    """Log-normal CV (%) from variances on the log scale: sqrt(exp(var) - 1)."""
    return [float(math.sqrt(math.exp(v) - 1.0) * 100.0) for v in variances]


def parameter_uncertainty(fit: MAPFit) -> List[Dict[str, Optional[float]]]:
    """Prior and posterior CV (%) for each parameter that carries variability."""
    prior = cv_percent(np.diag(fit.model.omega))
    cov = fit.posterior_covariance if fit.levels else None
    post = cv_percent(np.diag(cov)) if cov is not None else [None] * len(prior)
    return [{"parameter": name, "prior_cv_pct": p, "posterior_cv_pct": q}
            for name, p, q in zip(fit.model.eta_params, prior, post)]


def _draws(fit: MAPFit, n: int, seed: int) -> Optional[List[Dict[str, float]]]:
    if fit.eta_covariance is None:
        return None
    return fit.parameter_samples(n=n, seed=seed)


def concentration_band(fit: MAPFit, doses: Sequence[Dose], times: Sequence[float],
                       n: int = BAND_SAMPLES, level: float = BAND_LEVEL,
                       seed: int = 0) -> Optional[Dict[str, List[float]]]:
    """Pointwise central ``level`` band of concentration over ``times``.

    Returns None when the fit has no usable covariance.
    """
    draws = _draws(fit, n, seed)
    if draws is None:
        return None
    profiles = np.array([model_from_params(p).concentrations(doses, times)
                         for p in draws])
    tail = (1.0 - level) / 2.0 * 100.0
    return {"low": np.percentile(profiles, tail, axis=0).tolist(),
            "high": np.percentile(profiles, 100.0 - tail, axis=0).tolist(),
            "level": level}


def exposure_distribution(fit: MAPFit, exposure: Exposure,
                          target_low: float, target_high: float,
                          peak_threshold_mg_l: float,
                          n_auc: int = AUC_SAMPLES, n_peak: int = PEAK_SAMPLES,
                          bin_width: float = HISTOGRAM_BIN_WIDTH,
                          seed: int = 0) -> Optional[Dict[str, object]]:
    """Distribution of steady-state AUC24 and day-``EVALUATION_DAY`` peak.

    AUC24 = total daily dose / CL needs only clearance, so it is sampled densely;
    the peak needs a solve per draw and is sampled more sparsely. The histogram
    is returned already binned so that a front end draws it without the samples.
    """
    if fit.eta_covariance is None:
        return None
    cls = np.array([p["CL"] for p in fit.parameter_samples(n=n_auc, seed=seed)])
    aucs = exposure.total_daily_dose_mg / cls

    doses, t0, _ = _regimen_doses(exposure.dose_mg, exposure.interval_h,
                                  exposure.infusion_h, exposure.day)
    t_peak = [t0 + exposure.infusion_h]          # an infusion peaks as it ends
    peaks = np.array([model_from_params(p).concentrations(doses, t_peak)[0]
                      for p in fit.parameter_samples(n=n_peak, seed=seed + 1)])

    top = max(1000.0, float(np.percentile(aucs, 99.5)))
    top = math.ceil(top / bin_width) * bin_width
    counts, edges = np.histogram(aucs, bins=np.arange(0.0, top + bin_width, bin_width))
    return {
        "prob_below_target": float(np.mean(aucs < target_low)),
        "prob_in_target": float(np.mean((aucs >= target_low) & (aucs <= target_high))),
        "prob_above_target": float(np.mean(aucs > target_high)),
        "prob_peak_above": float(np.mean(peaks > peak_threshold_mg_l)),
        "peak_threshold_mg_l": float(peak_threshold_mg_l),
        "auc24_mean": float(np.mean(aucs)),
        "auc24_median": float(np.median(aucs)),
        "peak_median": float(np.median(peaks)),
        "histogram": {"bin_edges": edges.tolist(),
                      "density": (counts / counts.sum()).tolist()},
        "n_auc_samples": int(n_auc), "n_peak_samples": int(n_peak),
    }


def forecast(fit: MAPFit, dose_mg: float, interval_h: float, infusion_h: float,
             horizon_h: float = 72.0, n_points: int = 800,
             band: bool = True, seed: int = 0,
             start_h: Optional[float] = None) -> Dict[str, object]:
    """The recorded history followed by a new regimen, from the first dose onward.

    The new regimen starts at ``start_h`` -- when the next dose is due on the
    current schedule -- or, without it, one new interval after the last recorded
    dose. The forecast answers "what happens if we switch to this at the next dose".
    """
    history = sorted(fit.doses, key=lambda d: d.start_h)
    last = history[-1]
    start = start_h if start_h is not None else last.start_h + interval_h
    future = [Dose(amount_mg=dose_mg, start_h=start + i * interval_h, duration_h=infusion_h)
              for i in range(int(math.ceil(horizon_h / interval_h)) + 1)]
    end = last.start_h + horizon_h
    future = [d for d in future if d.start_h < end]
    doses = history + future
    times = time_grid(history[0].start_h, end, doses, n_points)
    out: Dict[str, object] = {
        "dose_mg": float(dose_mg), "interval_h": float(interval_h),
        "infusion_h": float(infusion_h), "forecast_start_h": float(last.start_h),
        "time_h": times.tolist(),
        "conc": fit.pk.concentrations(doses, times).tolist(),
        "dose_times_h": [d.start_h for d in doses],
        "next_dose_h": float(start),
        "low": None, "high": None,
    }
    if band:
        b = concentration_band(fit, doses, times, seed=seed)
        if b is not None:
            out["low"], out["high"] = b["low"], b["high"]
    return out


def steady_state_interval(fit: MAPFit, dose_mg: float, interval_h: float,
                          infusion_h: float, day: int = EVALUATION_DAY,
                          n_points: int = 400) -> Dict[str, List[float]]:
    """One dosing interval on day ``day`` of a regimen, timed from that dose."""
    doses, t0, t1 = _regimen_doses(dose_mg, interval_h, infusion_h, day)
    times = time_grid(t0, t1, doses, n_points)
    return {"time_into_interval_h": (times - t0).tolist(),
            "conc": fit.pk.concentrations(doses, times).tolist(),
            "day": int(day)}


def one_compartment_interval(sz: Dict[str, object], n_points: int = 200
                             ) -> Optional[Dict[str, List[float]]]:
    """The Sawchuk-Zaske one-compartment curve over its own dosing interval."""
    if not sz.get("available"):
        return None
    ke, vol = float(sz["ke"]), float(sz["volume_l"])
    tau, tinf = float(sz["interval_h"]), float(sz["infusion_h"])
    cmin, cmax = float(sz["cmin"]), float(sz["cmax"])
    rate = float(sz["dose_mg"]) / tinf
    t = np.unique(np.append(np.linspace(0.0, tau, n_points), tinf))   # the peak, exactly
    during = cmin * np.exp(-ke * t) + rate / (vol * ke) * (1.0 - np.exp(-ke * t))
    after = cmax * np.exp(-ke * (t - tinf))
    return {"time_into_interval_h": t.tolist(),
            "conc": np.where(t <= tinf, during, after).tolist()}
