"""A stable JSON contract for front ends.

One call in, one document out::

    from vankedoodle.api import evaluate
    response = evaluate(request_dict)

The request is plain JSON-compatible data -- no library objects -- and the
response is too, so the same contract serves a Streamlit app, a static
browser build, an HTTP service or an EHR integration. Front ends should read
values out of the response rather than recomputing anything.

Errors come back as data, not exceptions: a request that a clinician could
plausibly get wrong (a missing creatinine, a level before the first dose)
produces ``{"ok": false, "errors": [...]}`` with messages written to be shown
on screen. Only genuinely malformed input (a non-dict, a bad type) raises.

See API.md for the field-by-field contract and example_case.json for a
complete request.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ._version import __version__
from .bayes import fit as map_fit
from .core import Dose, DosingHistory, Level, Patient
from .dosing import (AUC_TARGET, AUC_TARGET_HIGH, AUC_TARGET_LOW, Exposure,
                     continuous_infusion, predict_exposure, recommend)
from .models import MODELS, get_model
from .reference import two_level_from_history
from .pk import time_grid
from .uncertainty import (concentration_band, exposure_distribution, forecast,
                          one_compartment_interval, parameter_uncertainty,
                          steady_state_interval)

SCHEMA_VERSION = "1.0"
DEFAULT_PEAK_THRESHOLD_MG_L = 60.0   # the peak ceiling recommend() already ranks against
DEFAULT_FORECAST_H = 72.0

_PATIENT_ALIASES = {
    "weight": "weight_kg", "wt": "weight_kg", "wt_kg": "weight_kg",
    "age": "age_years", "height": "height_cm", "ht_cm": "height_cm",
    "scr": "scr_mg_dl", "creatinine": "scr_mg_dl", "clcr": "crcl",
    "gestational_age": "gestational_age_weeks", "postnatal_age": "postnatal_age_days",
    "dob": "birthdate", "date_of_birth": "birthdate",
}
_PATIENT_FIELDS = {"weight_kg", "age_years", "height_cm", "scr_mg_dl", "scr_umol_l",
                   "crcl", "gestational_age_weeks", "postnatal_age_days", "sex",
                   "patient_id", "birthdate", "age_on"}


# ---------------------------------------------------------------- helpers
def _j(value):
    """Coerce numpy scalars and arrays to plain JSON-compatible Python."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_j(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _j(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_j(v) for v in value]
    return value


def _exposure_dict(e: Optional[Exposure], target_low: float,
                   target_high: float) -> Optional[Dict[str, Any]]:
    if e is None:
        return None
    return _j({
        "dose_mg": e.dose_mg, "interval_h": e.interval_h, "infusion_h": e.infusion_h,
        "total_daily_dose_mg": e.total_daily_dose_mg,
        "mg_per_kg_per_day": e.mg_per_kg_per_day,
        "auc24": e.auc24, "auc24_ci": list(e.auc24_ci) if e.auc24_ci else None,
        "prob_in_target": e.prob_in_target,
        "prob_below_target": e.prob_below_target,
        "prob_above_target": e.prob_above_target,
        "continuous": e.continuous,
        "in_target": bool(target_low <= e.auc24 <= target_high),
        "auc24_day": e.auc24_day, "evaluation_day": e.day,
        "cmax": e.cmax, "ctrough": e.ctrough,
        "flags": list(e.flags),
    })


def _build_patient(raw: Mapping, first_dose: Optional[_dt.datetime] = None) -> Patient:
    """A Patient from the request. A birthdate without ``age_on`` gives the age on
    the date of the first dose when doses carry clock times, otherwise today."""
    data = {}
    for key, value in dict(raw).items():
        key = _PATIENT_ALIASES.get(key, key)
        if key not in _PATIENT_FIELDS:
            raise ValueError(f"Unknown patient field {key!r}.")
        data[key] = value
    if data.get("birthdate") and not data.get("age_on") and first_dose is not None:
        data["age_on"] = first_dose.date()
    return Patient(**data)


def _build_history(request: Mapping) -> DosingHistory:
    if request.get("doses"):
        return DosingHistory.from_records(
            [{"amount_mg": d.get("amount_mg", d.get("dose_mg")),
              "time": d.get("time", d.get("start")),
              "duration_h": d.get("infusion_h", d.get("duration_h", 1.0))}
             for d in request["doses"]],
            origin=request.get("time_origin"))
    reg = request.get("regimen")
    if not reg:
        raise ValueError("Supply either 'doses' (a list) or 'regimen' (a repeating schedule).")
    return DosingHistory.repeating(
        dose_mg=float(reg["dose_mg"]), interval_h=float(reg["interval_h"]),
        n_doses=int(reg.get("n_doses", 1)), infusion_h=float(reg.get("infusion_h", 1.0)),
        start=reg.get("start", 0.0), first_dose_mg=reg.get("first_dose_mg"))


def _observed_interval(history: DosingHistory) -> float:
    starts = sorted(d.start_h for d in history.doses)
    gaps = [b - a for a, b in zip(starts[:-1], starts[1:]) if b > a]
    return min(gaps) if gaps else 24.0


def _series(fit, prior_fit, history: DosingHistory, recommended: Optional[Exposure],
            n_points: int = 600, project_h: float = 24.0,
            bands: bool = False, until_h: Optional[float] = None) -> Dict[str, Any]:
    """Concentration-time curves for plotting: history, and a forward projection.

    The projection begins at the last recorded dose rather than after it, so it
    overlaps the final infusion and the two curves join into one continuous line.
    With ``bands`` the prior and posterior curves carry 95% bands (``None`` where
    a fit has no usable covariance). ``until_h`` carries the history curves on to
    that time, normally when the next dose is due, so a forecast that starts
    there joins them without a gap.
    """
    t0 = min(d.start_h for d in history.doses)
    t_end = max(d.end_h for d in history.doses)
    if fit.levels:      # carry the curves past the last level so it sits inside them
        t_end = max(t_end, max(l.time_h for l in fit.levels) + 1.0)
    if until_h is not None:
        t_end = max(t_end, float(until_h))
    grid = time_grid(t0, t_end + 1e-9, history.doses, n_points)
    out = {"time_h": _j(grid),
           "conc_posterior": _j(fit.pk.concentrations(history.doses, grid)),
           "conc_prior": _j(prior_fit.pk.concentrations(history.doses, grid))}
    if bands:
        out["band_prior"] = _j(concentration_band(prior_fit, history.doses, grid))
        out["band_posterior"] = (_j(concentration_band(fit, history.doses, grid))
                                 if fit.levels else None)
    if recommended is not None:
        tau, amt = recommended.interval_h, recommended.dose_mg
        last = max(history.doses, key=lambda d: d.start_h)
        future = list(history.doses)
        t = last.start_h + tau
        while t < last.start_h + tau + project_h:
            future.append(Dose(amount_mg=amt, start_h=t, duration_h=recommended.infusion_h))
            t += tau
        pgrid = time_grid(last.start_h, last.start_h + tau + project_h, future, n_points)
        out["projection_time_h"] = _j(pgrid)
        out["projection_conc"] = _j(fit.pk.concentrations(future, pgrid))
    return out


# ---------------------------------------------------------------- main entry
def evaluate(request: Mapping) -> Dict[str, Any]:
    """Run the engine over one case and return a JSON-compatible document."""
    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping (parsed JSON object).")

    options = dict(request.get("options") or {})
    target = float(options.get("target_auc24", AUC_TARGET))
    target_low = float(options.get("target_low", AUC_TARGET_LOW))
    target_high = float(options.get("target_high", AUC_TARGET_HIGH))
    want_series = bool(options.get("series", True))
    want_uncertainty = bool(options.get("uncertainty", False))
    want_continuous = bool(options.get("continuous_infusion", False))
    peak_threshold = float(options.get("peak_threshold_mg_l", DEFAULT_PEAK_THRESHOLD_MG_L))
    forecast_h = float(options.get("forecast_h", DEFAULT_FORECAST_H))
    dose_increment = options.get("dose_increment")
    dose_increment = float(dose_increment) if dose_increment else None

    requested = options.get("model", "smit2021")
    model_names = list(MODELS) if requested in ("all", "*") else (
        list(requested) if isinstance(requested, (list, tuple)) else [requested])

    errors: List[str] = []
    try:
        history = _build_history(request)
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"Doses: {exc}")
        history = None
    try:
        patient = _build_patient(request.get("patient") or {},
                                 history.origin if history is not None else None)
    except (ValueError, TypeError) as exc:
        errors.insert(0, f"Patient: {exc}")
        patient = None
    levels: List[Level] = []
    if history is not None:
        try:
            levels = history.levels(request.get("levels") or [])
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"Levels: {exc}")
    for name in model_names:
        if name not in MODELS:
            errors.append(f"Unknown model {name!r}. Available: {sorted(MODELS)}.")
    if errors:
        return {"schema_version": SCHEMA_VERSION, "ok": False, "errors": errors,
                "engine_version": __version__}

    tau_given = _observed_interval(history)
    last_dose = history.last_dose
    results = []
    for name in model_names:
        model = get_model(name)
        entry: Dict[str, Any] = {"model": model.name, "label": model.label,
                                 "n_compartments": model.n_compartments,
                                 "population": model.population,
                                 "reference": model.reference}
        try:
            fit = map_fit(patient, history.doses, levels, model=name)
            prior_fit = map_fit(patient, history.doses, [], model=name)
        except ValueError as exc:
            entry.update({"ok": False, "error": str(exc)})
            results.append(entry)
            continue

        current = predict_exposure(fit, last_dose.amount_mg, tau_given,
                                   last_dose.duration_h, target_low=target_low,
                                   target_high=target_high)
        rec = recommend(fit, target_auc24=target,
                        target_low=target_low, target_high=target_high,
                        interval_h=options.get("interval_h"),
                        dose_increment=dose_increment,
                        max_cmax_mg_l=peak_threshold)
        candidate = None
        cand = options.get("candidate_regimen")
        if cand:
            c_tau = float(cand["interval_h"])
            c_inf = cand.get("infusion_h")
            candidate = predict_exposure(          # an infusion filling its interval is continuous
                fit, float(cand["dose_mg"]), c_tau, c_inf,
                target_low=target_low, target_high=target_high,
                continuous=c_inf is not None and float(c_inf) >= c_tau)

        residuals = fit.residuals() if levels else {}
        entry.update({
            "ok": True, "error": None,
            "warnings": list(fit.warnings),
            "out_of_population": any("extrapolation" in w for w in fit.warnings),
            "implausible_fit": bool(fit.implausible),
            "n_levels_used": len(levels),
            "posterior_method": fit.posterior_method,
            "posterior_ess": _j(fit.posterior_ess),
            "parameters": _j(fit.parameters),
            "population_parameters": _j(fit.population_parameters),
            "clearance_ml_per_h_per_kg": _j(fit.cl_per_kg * 1000.0),
            "eta": _j(fit.eta),
            "fit_to_levels": _j({k: list(v) for k, v in residuals.items()}),
            "current": _exposure_dict(current, target_low, target_high),
            "recommendation": _exposure_dict(rec.best, target_low, target_high),
            "alternatives": [_exposure_dict(e, target_low, target_high)
                             for e in rec.options[1:]],
            "candidate": _exposure_dict(candidate, target_low, target_high),
            "continuous_infusion": None,
            "series": (_series(fit, prior_fit, history, rec.best, bands=want_uncertainty,
                               until_h=last_dose.start_h + tau_given)
                       if want_series else None),
        })
        infusion = None
        if want_continuous:
            infusion = continuous_infusion(fit, target_auc24=target,
                                           dose_increment=dose_increment,
                                           target_low=target_low, target_high=target_high)
            entry["continuous_infusion"] = _exposure_dict(infusion, target_low, target_high)
        if want_uncertainty:
            def dist(e):
                return _j(exposure_distribution(fit, e, target_low, target_high,
                                                peak_threshold))
            entry["recommendation"]["distribution"] = dist(rec.best)
            for d, e in zip(entry["alternatives"], rec.options[1:]):
                d["distribution"] = dist(e)
            if candidate is not None:
                entry["candidate"]["distribution"] = dist(candidate)
            if infusion is not None:
                entry["continuous_infusion"]["distribution"] = dist(infusion)
            shown = candidate if candidate is not None else rec.best
            entry["forecast"] = _j({
                "regimen": "candidate" if candidate is not None else "recommendation",
                **forecast(fit, shown.dose_mg, shown.interval_h, shown.infusion_h,
                           horizon_h=forecast_h,
                           start_h=last_dose.start_h + tau_given)})
            entry["parameter_uncertainty"] = _j(parameter_uncertainty(fit))
            entry["steady_state_interval"] = _j(steady_state_interval(
                fit, last_dose.amount_mg, tau_given, last_dose.duration_h))
        results.append(entry)

    sz = _j(two_level_from_history(history.doses, levels, interval_h=tau_given))
    if sz["available"]:
        sz["curve"] = _j(one_compartment_interval(sz))
    comparison = {"sawchuk_zaske_auc24": sz.get("auc24") if sz["available"] else None,
                  "bayesian_auc24": {}, "percent_difference": {}}
    for entry in results:
        if not entry.get("ok"):
            continue
        auc = entry["current"]["auc24"]
        comparison["bayesian_auc24"][entry["model"]] = auc
        if sz["available"]:
            comparison["percent_difference"][entry["model"]] = _j(
                100.0 * (auc - sz["auc24"]) / sz["auc24"])

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "errors": [],
        "engine_version": __version__,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "case_id": request.get("case_id"),
        "patient": _j({
            "weight_kg": patient.weight_kg, "age_years": patient.age_years,
            "height_cm": patient.height_cm, "sex": patient.sex,
            "scr_mg_dl": patient.scr_mg_dl, "scr_umol_l": patient.scr_umol_l,
            "crcl_bedside_schwartz": patient.crcl, "pma_weeks": patient.pma_weeks,
            "gestational_age_weeks": patient.gestational_age_weeks,
            "postnatal_age_days": patient.postnatal_age_days,
            "birthdate": patient.birthdate.isoformat() if patient.birthdate else None,
            "age_on": patient.age_on.isoformat() if patient.age_on else None,
            "assumed_term_gestation": patient.assumed_term_gestation,
        }),
        "regimen_given": _j({
            "doses": [{"amount_mg": d.amount_mg, "time_h": d.start_h,
                       "infusion_h": d.duration_h} for d in history.doses],
            "n_doses": len(history.doses),
            "observed_interval_h": tau_given,
            "last_dose_mg": last_dose.amount_mg,
            "total_daily_dose_mg": last_dose.amount_mg * 24.0 / tau_given,
            "mg_per_kg_per_day": last_dose.amount_mg * 24.0 / tau_given / patient.weight_kg,
            "time_origin": history.origin.isoformat() if history.origin else None,
        }),
        "levels": _j([{"time_h": l.time_h, "conc_mg_l": l.conc_mg_l} for l in levels]),
        "target": {"auc24": target, "low": target_low, "high": target_high,
                   "peak_threshold_mg_l": peak_threshold},
        "results": results,
        "sawchuk_zaske": sz,
        "comparison": _j(comparison),
    }


# ---------------------------------------------------------------- study capture
CASE_RECORD_FIELDS = (
    "case_id", "recorded_at", "engine_version", "schema_version",
    "weight_kg", "age_years", "height_cm", "sex", "scr_mg_dl",
    "crcl_bedside_schwartz", "pma_weeks",
    "n_doses", "last_dose_mg", "observed_interval_h", "total_daily_dose_mg",
    "mg_per_kg_per_day", "n_levels", "level_times_h", "level_concs_mg_l",
    "model", "out_of_population", "implausible_fit", "clearance_l_per_h",
    "bayesian_auc24", "bayesian_auc24_low", "bayesian_auc24_high",
    "bayesian_in_target", "recommended_dose_mg", "recommended_interval_h",
    "recommended_auc24",
    "sawchuk_zaske_available", "sawchuk_zaske_reason", "sawchuk_zaske_auc24",
    "sawchuk_zaske_clearance_l_per_h", "percent_difference_vs_sawchuk_zaske",
)


def case_record(response: Mapping, model: Optional[str] = None,
                extra: Optional[Mapping] = None) -> Dict[str, Any]:
    """Flatten one evaluation into a single row for the study database.

    One row per case per model, with the fields Aims 1 and 3 need: both AUC
    estimates, their difference, how many levels were used, and the regimen.
    ``extra`` is merged in unchanged, for study identifiers or notes.
    """
    if not response.get("ok"):
        raise ValueError(f"Cannot record a failed evaluation: {response.get('errors')}")
    usable = [r for r in response["results"] if r.get("ok")]
    if not usable:
        raise ValueError("No model produced a result for this case.")
    entry = next((r for r in usable if r["model"] == model), usable[0])

    sz = response["sawchuk_zaske"]
    ci = entry["current"].get("auc24_ci") or [None, None]
    row = {
        "case_id": response.get("case_id"),
        "recorded_at": response.get("generated_at"),
        "engine_version": response.get("engine_version"),
        "schema_version": response.get("schema_version"),
        "n_levels": len(response["levels"]),
        "level_times_h": [l["time_h"] for l in response["levels"]],
        "level_concs_mg_l": [l["conc_mg_l"] for l in response["levels"]],
        "model": entry["model"],
        "out_of_population": entry["out_of_population"],
        "implausible_fit": entry.get("implausible_fit", False),
        "clearance_l_per_h": entry["parameters"]["CL"],
        "bayesian_auc24": entry["current"]["auc24"],
        "bayesian_auc24_low": ci[0],
        "bayesian_auc24_high": ci[1],
        "bayesian_in_target": entry["current"]["in_target"],
        "recommended_dose_mg": entry["recommendation"]["dose_mg"],
        "recommended_interval_h": entry["recommendation"]["interval_h"],
        "recommended_auc24": entry["recommendation"]["auc24"],
        "sawchuk_zaske_available": sz["available"],
        "sawchuk_zaske_reason": sz.get("reason"),
        "sawchuk_zaske_auc24": sz.get("auc24"),
        "sawchuk_zaske_clearance_l_per_h": sz.get("clearance_l_h"),
        "percent_difference_vs_sawchuk_zaske":
            response["comparison"]["percent_difference"].get(entry["model"]),
    }
    for key in ("weight_kg", "age_years", "height_cm", "sex", "scr_mg_dl",
                "crcl_bedside_schwartz", "pma_weeks"):
        row[key] = response["patient"].get(key)
    for key in ("n_doses", "last_dose_mg", "observed_interval_h",
                "total_daily_dose_mg", "mg_per_kg_per_day"):
        row[key] = response["regimen_given"].get(key)
    if extra:
        row.update(dict(extra))
    return row


def append_case(path, record: Mapping) -> None:
    """Append one case record to a JSON Lines file, creating it if needed."""
    import json
    import pathlib
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(_j(dict(record)), default=str) + "\n")


def load_cases(path) -> List[Dict[str, Any]]:
    """Read a JSON Lines case file back."""
    import json
    import pathlib
    target = pathlib.Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf8").splitlines()
            if line.strip()]
