"""Exposure metrics and dose selection.

Everything here works off either a fitted individual (:class:`~vankedoodle.bayes.MAPFit`)
or the population prior, so the same functions serve empiric starting doses and
post-level dose corrections.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .bayes import MAPFit, fit as map_fit
from .core import Dose, DosingHistory, Level, Patient
from .models import PopPKModel, get_model
from .pk import CompartmentalModel, model_from_params

AUC_TARGET_LOW = 400.0     # mg*h/L, 2020 consensus guideline AUC/MIC target (MIC = 1)
AUC_TARGET_HIGH = 600.0
AUC_TARGET = 500.0

MAX_INFUSION_RATE_MG_PER_H = 600.0   # 10 mg/min, the usual infusion-rate ceiling
MG_PER_KG_PER_DAY_ALERT = 100.0      # flag (not cap) above this weight-normalised daily dose
EVALUATION_DAY = 3                   # peak/trough and day AUC are reported on this day


def default_interval(patient: Patient) -> float:
    """Dosing interval conventionally used in children, by weight and renal function.

    Weight bands follow the pediatric guideline in Smit et al. 2021 Table III
    (<30 kg q6h, 30-70 kg q8h, >70 kg q12h), lengthened when renal function is
    markedly reduced.  The interval is a clinical convention, not a model output:
    different population models parameterise the peripheral compartment very
    differently, so deriving it from a model's terminal half-life is not robust.
    Override with ``interval_h`` anywhere this is used.
    """
    crcl = patient.crcl
    if crcl is not None:
        if crcl < 15.0:
            return 24.0
        if crcl < 30.0:
            return 12.0
    if patient.weight_kg < 30.0:
        return 6.0
    if patient.weight_kg <= 70.0:
        return 8.0
    return 12.0


def infusion_duration(dose_mg: float, minimum_h: float = 1.0,
                      max_rate_mg_per_h: float = MAX_INFUSION_RATE_MG_PER_H) -> float:
    """Infusion length respecting a maximum infusion rate, rounded up to 0.5 h."""
    need = dose_mg / max_rate_mg_per_h
    return float(max(minimum_h, math.ceil(need * 2) / 2))


def round_dose(dose_mg: float, increment: Optional[float] = None) -> float:
    """Round to a practically measurable amount."""
    if increment is None:
        increment = 5.0 if dose_mg < 100 else (25.0 if dose_mg < 1000 else 50.0)
    return float(max(increment, round(dose_mg / increment) * increment))


@dataclass
class Exposure:
    """Predicted exposure for a steady-state regimen."""
    dose_mg: float
    interval_h: float
    infusion_h: float
    total_daily_dose_mg: float
    mg_per_kg_per_day: float
    auc24: float                 # steady-state AUC24 = total daily dose / CL
    cmax: float                  # peak on the evaluation day
    ctrough: float               # trough on the evaluation day
    auc24_day: Optional[float] = None   # actual AUC over the evaluation day
    day: int = EVALUATION_DAY
    auc24_ci: Optional[tuple] = None
    prob_in_target: Optional[float] = None
    flags: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def _regimen_doses(dose_mg: float, interval_h: float, infusion_h: float,
                   day: int = EVALUATION_DAY) -> tuple:
    """A dose train covering ``day`` days, plus the bounds of that day's last interval."""
    horizon = 24.0 * day
    n = int(math.ceil(horizon / interval_h)) + 1
    doses = [Dose(amount_mg=dose_mg, start_h=i * interval_h, duration_h=infusion_h)
             for i in range(n)]
    last_start = (math.ceil(horizon / interval_h) - 1) * interval_h
    return doses, last_start, last_start + interval_h


def predict_exposure(source, dose_mg: float, interval_h: float,
                     infusion_h: Optional[float] = None,
                     ci: bool = True, n_samples: int = 2000,
                     target_low: float = AUC_TARGET_LOW,
                     target_high: float = AUC_TARGET_HIGH,
                     day: int = EVALUATION_DAY,
                     weight_kg: Optional[float] = None) -> Exposure:
    """Exposure for a candidate regimen.

    ``auc24`` is the steady-state value (total daily dose / CL, exact for any
    linear model); ``auc24_day``, ``cmax`` and ``ctrough`` are the actual values
    on day ``day`` of therapy, which is what a clinician sees and which does not
    depend on how deep a peripheral compartment a given model happens to carry.

    ``source`` is a :class:`MAPFit` (individual parameters, with uncertainty) or a
    dict of CL/V1/Q/V2.
    """
    if isinstance(source, MAPFit):
        pars = source.parameters
        weight_kg = weight_kg or source.patient.weight_kg
    else:
        pars = dict(source)
    pk = model_from_params(pars)

    if infusion_h is None:
        infusion_h = infusion_duration(dose_mg)
    if infusion_h >= interval_h:
        infusion_h = max(0.5, interval_h - 0.5)

    tdd = dose_mg * 24.0 / interval_h
    auc24 = pk.auc24_steady_state(tdd)

    doses, t0, t1 = _regimen_doses(dose_mg, interval_h, infusion_h, day)
    grid = np.linspace(t0, t1, 241)
    conc, _ = pk.simulate(doses, grid)
    cmax = float(np.nanmax(conc))
    ctrough = float(conc[-1])
    auc24_day = pk.auc(doses, 24.0 * (day - 1), 24.0 * day)

    auc_ci = None
    p_target = None
    if ci and isinstance(source, MAPFit) and source.eta_covariance is not None:
        cls = np.array([p["CL"] for p in source.parameter_samples(n=n_samples)])
        aucs = tdd / cls
        auc_ci = (float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95)))
        p_target = float(np.mean((aucs >= target_low) & (aucs <= target_high)))

    flags = []
    if weight_kg and tdd / weight_kg > MG_PER_KG_PER_DAY_ALERT:
        flags.append(
            f"{tdd/weight_kg:.0f} mg/kg/day exceeds the usual pediatric ceiling of "
            f"{MG_PER_KG_PER_DAY_ALERT:.0f} mg/kg/day; confirm against local policy.")
    if auc24 < target_low:
        flags.append(f"Predicted AUC24 {auc24:.0f} is below the {target_low:.0f} target.")
    if auc24 > target_high:
        flags.append(f"Predicted AUC24 {auc24:.0f} exceeds the {target_high:.0f} target.")
    if ctrough > 20:
        flags.append(f"Predicted day-{day} trough {ctrough:.1f} mg/L is above 20 mg/L.")

    return Exposure(dose_mg=float(dose_mg), interval_h=float(interval_h),
                    infusion_h=float(infusion_h), total_daily_dose_mg=float(tdd),
                    mg_per_kg_per_day=float(tdd / weight_kg) if weight_kg else float("nan"),
                    auc24=float(auc24), cmax=cmax, ctrough=ctrough,
                    auc24_day=float(auc24_day), day=int(day),
                    auc24_ci=auc_ci, prob_in_target=p_target, flags=flags)


@dataclass
class Recommendation:
    """A recommended regimen plus the alternatives that were considered."""
    best: Exposure
    options: list
    target_auc24: float
    basis: str                      # "individual (Bayesian)" or "population prior"
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        e = self.best
        lines = [
            f"Recommended: {e.dose_mg:g} mg IV every {e.interval_h:g} h "
            f"over {e.infusion_h:g} h "
            f"({e.total_daily_dose_mg:g} mg/day, {e.mg_per_kg_per_day:.1f} mg/kg/day)",
            f"Predicted steady-state AUC24 = {e.auc24:.0f} mg*h/L"
            + (f" (90% CI {e.auc24_ci[0]:.0f}-{e.auc24_ci[1]:.0f})" if e.auc24_ci else "")
            + (f"; P(AUC24 in target) = {e.prob_in_target*100:.0f}%"
               if e.prob_in_target is not None else ""),
            f"Day-{e.day} peak {e.cmax:.1f} mg/L, trough {e.ctrough:.1f} mg/L"
            + (f", AUC over day {e.day} = {e.auc24_day:.0f} mg*h/L"
               if e.auc24_day is not None else ""),
            f"Basis: {self.basis}",
        ]
        return "\n".join(lines + [f"! {w}" for w in (self.warnings + e.flags)])


def recommend(source,
              target_auc24: float = AUC_TARGET,
              intervals: Sequence[float] = (6.0, 8.0, 12.0, 24.0),
              dose_increment: Optional[float] = None,
              max_dose_mg: float = 2000.0,
              max_daily_dose_mg: float = 4500.0,
              max_cmax_mg_l: float = 60.0,
              min_infusion_h: float = 1.0,
              interval_h: Optional[float] = None,
              target_low: float = AUC_TARGET_LOW,
              target_high: float = AUC_TARGET_HIGH,
              ci: bool = True,
              weight_kg: Optional[float] = None) -> Recommendation:
    """Choose a dose and interval that put the predicted AUC24 on target.

    The dose follows directly from AUC24,ss = total daily dose / CL; each
    candidate interval is then evaluated after rounding the dose to a practical
    amount.  Candidates are ranked by (i) landing inside the AUC target window,
    (ii) keeping the predicted peak below ``max_cmax_mg_l``, (iii) closeness to
    the interval conventionally used for that patient (:func:`default_interval`),
    and (iv) closeness to the target AUC.  Pass ``interval_h`` to fix the
    interval instead.

    ``max_dose_mg``, ``max_daily_dose_mg`` and ``max_cmax_mg_l`` are
    institutional ceilings, not model outputs; override them to match local policy.
    """
    if isinstance(source, MAPFit):
        cl = source.parameters["CL"]
        wt = weight_kg or source.patient.weight_kg
        basis = ("individual (Bayesian MAP, "
                 f"{len(source.levels)} level{'s' if len(source.levels) != 1 else ''})"
                 if source.levels else "population prior (no levels)")
        warns = list(source.warnings)
    else:
        cl = float(dict(source)["CL"])
        wt = weight_kg
        basis = "supplied parameters"
        warns = []

    tdd_target = target_auc24 * cl
    preferred = (float(interval_h) if interval_h is not None
                 else (default_interval(source.patient) if isinstance(source, MAPFit)
                       else 8.0))
    if interval_h is not None:
        intervals = (float(interval_h),)
    options = []
    for tau in intervals:
        raw = tdd_target * tau / 24.0
        dose = round_dose(min(raw, max_dose_mg), dose_increment)
        if dose * 24.0 / tau > max_daily_dose_mg:
            dose = round_dose(max_daily_dose_mg * tau / 24.0, dose_increment)
        inf_h = infusion_duration(dose, minimum_h=min_infusion_h)
        exp = predict_exposure(source, dose, tau, inf_h, ci=ci,
                               target_low=target_low, target_high=target_high,
                               weight_kg=wt)
        if raw > max_dose_mg:
            exp.flags.append(f"Per-dose ceiling of {max_dose_mg:g} mg applied.")
        if tdd_target > max_daily_dose_mg:
            exp.flags.append(f"Daily-dose ceiling of {max_daily_dose_mg:g} mg applied.")
        options.append(exp)

        if exp.cmax > max_cmax_mg_l:
            exp.flags.append(
                f"Predicted peak {exp.cmax:.0f} mg/L exceeds {max_cmax_mg_l:.0f} mg/L; "
                "a shorter interval is preferable.")

    def rank(e: Exposure):
        on_target = target_low <= e.auc24 <= target_high
        peak_ok = e.cmax <= max_cmax_mg_l
        tau_fit = abs(math.log2(e.interval_h / preferred))
        return (not on_target, not peak_ok, tau_fit,
                abs(e.auc24 - target_auc24) / target_auc24)

    options_sorted = sorted(options, key=rank)
    return Recommendation(best=options_sorted[0], options=options_sorted,
                          target_auc24=target_auc24, basis=basis, warnings=warns)


def initial_dose(patient: Patient, target_auc24: float = AUC_TARGET,
                 model: str | PopPKModel = "smit2021", **kwargs) -> Recommendation:
    """Empiric starting regimen from the population model alone (no levels yet)."""
    f = map_fit(patient, doses=[], levels=[], model=model)
    return recommend(f, target_auc24=target_auc24, **kwargs)


def auc_over_window(f: MAPFit, start_h: float = 0.0, width_h: float = 24.0) -> float:
    """Actual (not steady-state) AUC over a window of the recorded dosing history."""
    return f.auc(start_h, start_h + width_h)


def exposure_timeline(f: MAPFit, until_h: Optional[float] = None,
                      step_h: float = 0.25) -> Dict[str, np.ndarray]:
    """Concentration-time profile and running 24-h AUC for the recorded history."""
    if until_h is None:
        last = f.doses[-1] if f.doses else None
        until_h = (last.start_h + 24.0) if last else 24.0
    t0 = min([d.start_h for d in f.doses], default=0.0)
    times = np.arange(t0, until_h + step_h, step_h)
    conc, cum = f.pk.simulate(f.doses, times)
    auc24 = np.full(times.shape, np.nan)
    for i, t in enumerate(times):
        if t - t0 >= 24.0:
            j = int(np.searchsorted(times, t - 24.0))
            auc24[i] = cum[i] - cum[j]
    return {"time_h": times, "conc_mg_l": conc, "cum_auc": cum, "rolling_auc24": auc24}


# --------------------------------------------------------------------------
# Published dosing guideline (Smit et al. 2021, Table III)
# --------------------------------------------------------------------------
SMIT_DOSE_CAP_MG_PER_DAY = 5500.0   # the paper recommends carrying the obese-adult cap over


def smit_nomogram(patient: Patient, loading_dose_mg_per_kg: float = 15.0) -> Dict[str, float]:
    """The guideline table from Smit et al. 2021 (Table III), for children 1-18 y.

    Returned as mg/kg per dose and the resulting mg amounts.  This is the
    published fixed guideline, provided as an independent cross-check and as a
    starting regimen when no model-based calculation is wanted; it is not used
    by the Bayesian engine.
    """
    wt = patient.weight_kg
    crcl = float(patient.crcl)
    if wt < 30:
        col = 0
    elif wt <= 70:
        col = 1
    else:
        col = 2
    table = [
        # (crcl lower bound, [per-dose mg/kg by weight column], [interval by column], loading?)
        (90.0, [15.0, 15.0, 18.0], [6.0, 8.0, 12.0], False),
        (50.0, [11.0, 11.0, 12.0], [6.0, 8.0, 12.0], True),
        (30.0, [5.0, 5.0, 6.0], [6.0, 8.0, 12.0], True),
        (10.0, [5.0, 3.0, 3.0], [12.0, 12.0, 12.0], True),
    ]
    for lower, mgkg, taus, loading in table:
        if crcl > lower:
            per_dose = mgkg[col]
            tau = taus[col]
            break
    else:
        raise ValueError(
            f"Bedside Schwartz CLcr {crcl:.1f} mL/min/1.73 m2 is below the range of the "
            "published guideline (10 mL/min/1.73 m2)."
        )
    dose_mg = per_dose * wt
    tdd = dose_mg * 24.0 / tau
    capped = False
    if tdd > SMIT_DOSE_CAP_MG_PER_DAY:
        dose_mg = SMIT_DOSE_CAP_MG_PER_DAY * tau / 24.0
        tdd = SMIT_DOSE_CAP_MG_PER_DAY
        capped = True
    return {"mg_per_kg_per_dose": per_dose, "interval_h": tau,
            "dose_mg": dose_mg, "total_daily_dose_mg": tdd,
            "first_dose_mg": loading_dose_mg_per_kg * wt if loading else dose_mg,
            "loading_dose_applied": loading, "dose_cap_applied": capped}
