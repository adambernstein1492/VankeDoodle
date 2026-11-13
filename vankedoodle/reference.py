"""Reference (non-Bayesian) AUC estimation: the Sawchuk-Zaske two-level method.

This is the current standard of care at the study site and the comparator for
Aim 1.  Two concentrations drawn within one dosing interval at steady state are
used to derive a first-order elimination rate constant; the AUC over the
interval is then the trapezoid during the infusion plus the log-linear area
during elimination (Pai MP et al., Adv Drug Deliv Rev 2014;77:50-57).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import math


@dataclass
class SawchukZaskeResult:
    ke: float                 # 1/h
    half_life_h: float
    volume_l: float
    clearance_l_h: float
    cmax: float               # at end of infusion
    cmin: float               # at end of the dosing interval
    auc_tau: float
    auc24: float

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def sawchuk_zaske(dose_mg: float, interval_h: float, infusion_h: float,
                  t1_h: float, c1_mg_l: float,
                  t2_h: float, c2_mg_l: float) -> SawchukZaskeResult:
    """Two-level steady-state AUC.

    Times ``t1_h`` and ``t2_h`` are hours from the *start* of the infusion of the
    dose during which the levels were drawn; ``t1_h`` must be after the end of
    the infusion (post-distribution) and ``t2_h`` after ``t1_h``.
    """
    if t2_h <= t1_h:
        raise ValueError("The second level must be drawn after the first.")
    if t1_h < infusion_h:
        raise ValueError(
            "The first level must be drawn after the end of the infusion; "
            "the method assumes both levels lie in the log-linear elimination phase."
        )
    if c1_mg_l <= 0 or c2_mg_l <= 0:
        raise ValueError("Concentrations must be > 0 mg/L.")
    if c2_mg_l >= c1_mg_l:
        raise ValueError(
            "The second concentration must be lower than the first for a "
            "first-order elimination rate to be estimated."
        )

    ke = math.log(c1_mg_l / c2_mg_l) / (t2_h - t1_h)
    t_half = math.log(2.0) / ke

    cmax = c1_mg_l * math.exp(ke * (t1_h - infusion_h))      # back-extrapolate to end of infusion
    cmin = c2_mg_l * math.exp(-ke * (interval_h - t2_h))     # forward to end of interval
    c_pre = cmin                                             # steady state: trough precedes the dose

    auc_infusion = 0.5 * (c_pre + cmax) * infusion_h
    auc_elimination = (cmax - cmin) / ke
    auc_tau = auc_infusion + auc_elimination

    rate = dose_mg / infusion_h
    volume = (rate / ke) * (1.0 - math.exp(-ke * infusion_h)) / (
        cmax - c_pre * math.exp(-ke * infusion_h))
    clearance = ke * volume

    return SawchukZaskeResult(ke=ke, half_life_h=t_half, volume_l=volume,
                              clearance_l_h=clearance, cmax=cmax, cmin=cmin,
                              auc_tau=auc_tau, auc24=auc_tau * 24.0 / interval_h)


def mpe_mape(reference, test) -> Dict[str, float]:
    """Mean percentage error (bias) and mean absolute percentage error (precision).

    Errors are expressed relative to ``reference``, matching the study protocol.
    """
    ref = [float(r) for r in reference]
    tst = [float(t) for t in test]
    if len(ref) != len(tst) or not ref:
        raise ValueError("reference and test must be non-empty and the same length.")
    pe = [100.0 * (t - r) / r for r, t in zip(ref, tst)]
    n = len(pe)
    return {"n": n,
            "mpe": sum(pe) / n,
            "mape": sum(abs(p) for p in pe) / n,
            "percent_errors": pe}


def two_level_from_history(doses: Sequence, levels: Sequence,
                           interval_h: Optional[float] = None) -> Dict[str, object]:
    """Find a usable two-level pair in a dosing history and apply Sawchuk-Zaske.

    The method needs two concentrations drawn within one dosing interval, the
    first after the end of that infusion.  This picks the most recent interval
    that satisfies those conditions, so a front end can simply hand over the
    whole case and be told either the result or why it cannot be computed.

    Returns a dict with ``available`` and, when False, a ``reason`` written for
    a clinician rather than a programmer.
    """
    doses = sorted(doses, key=lambda d: d.start_h)
    levels = sorted(levels, key=lambda l: l.time_h)
    if len(levels) < 2:
        return {"available": False,
                "reason": "Two levels are needed; only "
                          f"{len(levels)} {'was' if len(levels) == 1 else 'were'} supplied."}
    if not doses:
        return {"available": False, "reason": "No doses were supplied."}

    starts = [d.start_h for d in doses]
    gaps = [b - a for a, b in zip(starts[:-1], starts[1:]) if b > a]
    default_tau = interval_h if interval_h else (min(gaps) if gaps else 24.0)

    attempted = False
    for i in range(len(doses) - 1, -1, -1):
        dose = doses[i]
        tau = (starts[i + 1] - starts[i]) if i + 1 < len(doses) else default_tau
        window = [l for l in levels if dose.start_h <= l.time_h < dose.start_h + tau]
        if len(window) < 2:
            continue
        attempted = True
        first, last = window[0], window[-1]
        t1 = first.time_h - dose.start_h
        t2 = last.time_h - dose.start_h
        if t1 < dose.duration_h:
            continue
        if last.conc_mg_l >= first.conc_mg_l:
            continue
        result = sawchuk_zaske(dose_mg=dose.amount_mg, interval_h=tau,
                               infusion_h=dose.duration_h,
                               t1_h=t1, c1_mg_l=first.conc_mg_l,
                               t2_h=t2, c2_mg_l=last.conc_mg_l)
        out = {"available": True, "reason": None,
               "dose_mg": dose.amount_mg, "interval_h": tau,
               "infusion_h": dose.duration_h,
               "levels_used": [{"time_h": first.time_h, "hours_into_interval": t1,
                                "conc_mg_l": first.conc_mg_l},
                               {"time_h": last.time_h, "hours_into_interval": t2,
                                "conc_mg_l": last.conc_mg_l}]}
        out.update(result.as_dict())
        return out

    if attempted:
        return {"available": False,
                "reason": "Two levels fall in the same dosing interval, but the pair is "
                          "unusable: the first must be drawn after the end of the "
                          "infusion and the second must be lower than the first."}
    return {"available": False,
            "reason": "No two levels fall within the same dosing interval. The "
                      "two-level method needs both samples from one interval; the "
                      "Bayesian estimate does not."}
