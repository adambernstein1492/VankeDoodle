"""The two calls a bedside user actually needs.

``start_dose`` -- an empiric regimen before any level has been drawn.
``assess``     -- given the doses given and the level(s) drawn, what is the
                  patient's AUC24 now and what dose should follow.

Both take plain numbers, strings and dicts so that a web form, a CLI or a
notebook can call them without constructing model objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .bayes import MAPFit, fit as map_fit
from .core import Dose, DosingHistory, Level, Patient, TimeLike
from .dosing import (AUC_TARGET, AUC_TARGET_HIGH, AUC_TARGET_LOW, Exposure,
                     Recommendation, initial_dose, predict_exposure, recommend)
from .models import MODELS, PopPKModel, get_model


def _as_patient(obj: Union[Patient, Mapping]) -> Patient:
    if isinstance(obj, Patient):
        return obj
    d = dict(obj)
    alias = {"weight": "weight_kg", "wt": "weight_kg", "wt_kg": "weight_kg",
             "age": "age_years", "height": "height_cm", "ht_cm": "height_cm",
             "scr": "scr_mg_dl", "creatinine": "scr_mg_dl",
             "crcl_ml_min_173": "crcl", "clcr": "crcl"}
    for k, v in list(d.items()):
        if k in alias:
            d[alias[k]] = d.pop(k)
    return Patient(**d)


def _as_history(regimen: Union[DosingHistory, Mapping, Sequence]) -> DosingHistory:
    if isinstance(regimen, DosingHistory):
        return regimen
    if isinstance(regimen, Mapping):
        d = dict(regimen)
        return DosingHistory.repeating(
            dose_mg=float(d["dose_mg"]),
            interval_h=float(d["interval_h"]),
            n_doses=int(d.get("n_doses", 1)),
            infusion_h=float(d.get("infusion_h", 1.0)),
            start=d.get("start", 0.0),
            first_dose_mg=d.get("first_dose_mg"),
        )
    return DosingHistory.from_records(regimen)


@dataclass
class Assessment:
    """Everything a dosing decision needs, in one object."""
    fit: MAPFit
    current: Optional[Exposure]
    recommendation: Recommendation
    auc24_now: Optional[float]
    target: tuple = (AUC_TARGET_LOW, AUC_TARGET_HIGH)

    # convenient passthroughs
    @property
    def parameters(self):
        return self.fit.parameters

    @property
    def clearance(self) -> float:
        return self.fit.parameters["CL"]

    @property
    def in_target(self) -> Optional[bool]:
        if self.auc24_now is None:
            return None
        return self.target[0] <= self.auc24_now <= self.target[1]

    def summary(self) -> str:
        f = self.fit
        lines = [f"Patient: {f.patient.weight_kg:g} kg"
                 + (f", {f.patient.age_years:g} y" if f.patient.age_years else "")
                 + f", CLcr {f.patient.crcl:.0f} mL/min/1.73 m2",
                 f"Model: {f.model.name} | levels used: {len(f.levels)}",
                 f"Individual CL = {f.parameters['CL']:.2f} L/h "
                 f"({f.cl_per_kg*1000:.0f} mL/h/kg), V1 = {f.parameters['V1']:.1f} L, "
                 f"V2 = {f.parameters['V2']:.1f} L"]
        if self.auc24_now is not None:
            status = "in target" if self.in_target else "OFF TARGET"
            lines.append(f"AUC24 on the current regimen = {self.auc24_now:.0f} mg*h/L "
                         f"({status}, target {self.target[0]:.0f}-{self.target[1]:.0f})")
        if f.levels:
            r = f.residuals()
            pairs = ", ".join(f"{o:.1f} obs / {p:.1f} pred"
                              for o, p in zip(r["observed"], r["predicted"]))
            lines.append(f"Fit to levels: {pairs}")
        lines.append("")
        lines.append(self.recommendation.summary())
        return "\n".join(lines)


def start_dose(patient: Union[Patient, Mapping, None] = None,
               target_auc24: float = AUC_TARGET,
               model: str = "smit2021", **patient_kwargs) -> Recommendation:
    """Empiric regimen for a patient with no levels yet.

    >>> rec = start_dose(weight_kg=20, age_years=6, height_cm=115, scr=0.4)
    """
    pt = _as_patient(patient if patient is not None else patient_kwargs)
    return initial_dose(pt, target_auc24=target_auc24, model=model)


def assess(patient: Union[Patient, Mapping],
           regimen: Union[DosingHistory, Mapping, Sequence],
           levels: Iterable = (),
           target_auc24: float = AUC_TARGET,
           model: str = "smit2021",
           target_low: float = AUC_TARGET_LOW,
           target_high: float = AUC_TARGET_HIGH,
           **recommend_kwargs) -> Assessment:
    """Fit the patient's levels and recommend the next regimen.

    ``levels`` may be ``(time, concentration)`` pairs or dicts; times may be
    datetimes/strings (if the regimen was given with datetimes) or hours after
    the first dose.
    """
    pt = _as_patient(patient)
    hist = _as_history(regimen)
    lv = hist.levels(levels) if levels else []
    f = map_fit(pt, hist.doses, lv, model=model)

    current = None
    auc_now = None
    if hist.doses:
        last = hist.last_dose
        same = [d for d in hist.doses if abs(d.amount_mg - last.amount_mg) < 1e-9]
        if len(same) >= 2:
            tau = min(b.start_h - a.start_h for a, b in zip(same[:-1], same[1:])
                      if b.start_h > a.start_h)
        else:
            tau = 24.0
        current = predict_exposure(f, last.amount_mg, tau, last.duration_h,
                                   target_low=target_low, target_high=target_high)
        auc_now = current.auc24

    rec = recommend(f, target_auc24=target_auc24, target_low=target_low,
                    target_high=target_high, **recommend_kwargs)
    return Assessment(fit=f, current=current, recommendation=rec,
                      auc24_now=auc_now, target=(target_low, target_high))


def compare_models(patient: Union[Patient, Mapping],
                   regimen: Union[DosingHistory, Mapping, Sequence],
                   levels: Iterable = (),
                   models: Optional[Sequence[str]] = None,
                   target_auc24: float = AUC_TARGET) -> list:
    """Fit the same patient under several population models and tabulate the result.

    Returns one row per model with the individual clearance, the AUC24 on the
    current regimen, the fit to the measured levels and the recommended dose.
    Models whose required covariates are missing are reported with the reason
    rather than silently skipped.
    """
    pt = _as_patient(patient)
    names = list(models) if models is not None else list(MODELS)
    rows = []
    for nm in names:
        mdl = get_model(nm)
        row = {"model": mdl.name, "label": mdl.label,
               "n_compartments": mdl.n_compartments}
        try:
            a = assess(pt, regimen, levels=levels, target_auc24=target_auc24, model=nm)
        except (ValueError, KeyError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        r = a.fit.residuals() if a.fit.levels else {}
        row.update({
            "CL_L_per_h": a.clearance,
            "CL_mL_per_h_per_kg": a.fit.cl_per_kg * 1000.0,
            "auc24_current": a.auc24_now,
            "in_target": a.in_target,
            "predicted_levels": [float(x) for x in r.get("predicted", [])],
            "mean_percent_error": (float(np.mean(r["percent_error"]))
                                   if len(r.get("percent_error", [])) else None),
            "recommended_dose_mg": a.recommendation.best.dose_mg,
            "recommended_interval_h": a.recommendation.best.interval_h,
            "recommended_auc24": a.recommendation.best.auc24,
            "warnings": a.fit.warnings,
        })
        rows.append(row)
    return rows
