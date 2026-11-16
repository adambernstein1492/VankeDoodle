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


def _same_dose(a, b) -> bool:
    """Same amount and the same infusion length (to the precision charted)."""
    return (abs(a.amount_mg - b.amount_mg) <= max(0.5, 0.01 * a.amount_mg)
            and abs(a.duration_h - b.duration_h) <= 0.05)


def _own_dose(doses: Sequence, t: float) -> Optional[int]:
    """Index of the dose a level drawn at ``t`` follows: the last one started by then."""
    idx = None
    for i, d in enumerate(doses):
        if d.start_h <= t + 1e-9:
            idx = i
    return idx


def _pair_result(dose, tau, first, t1, last, t2, extra=None) -> Dict[str, object]:
    result = sawchuk_zaske(dose_mg=dose.amount_mg, interval_h=tau,
                           infusion_h=dose.duration_h,
                           t1_h=t1, c1_mg_l=first.conc_mg_l,
                           t2_h=t2, c2_mg_l=last.conc_mg_l)
    out = {"available": True, "reason": None, "extrapolated": False,
           "dose_mg": dose.amount_mg, "interval_h": tau,
           "infusion_h": dose.duration_h,
           "levels_used": [{"time_h": first.time_h, "hours_into_interval": t1,
                            "conc_mg_l": first.conc_mg_l},
                           {"time_h": last.time_h, "hours_into_interval": t2,
                            "conc_mg_l": last.conc_mg_l}]}
    out.update(extra or {})
    out.update(result.as_dict())
    return out


def _most_recent_pair(doses, levels, tau_regimen) -> Dict[str, object]:
    """Sawchuk-Zaske on the two most recent levels, placed in one interval.

    Each level is timed from the dose it follows. When both follow the same dose
    this is the classic same-interval calculation. When they follow different
    doses (typically a trough drawn just before dose N and a peak after it), the
    two are placed in one interval of the current regimen, which is valid at
    steady state only if every dose from the earlier one to the later one is the
    same amount over the same infusion, given at that interval. Those conditions
    are checked; any failure is returned as a reason, not silently ignored.
    """
    a, b = levels[-2], levels[-1]
    ia, ib = _own_dose(doses, a.time_h), _own_dose(doses, b.time_h)
    if ia is None or ib is None:
        return {"available": False,
                "reason": "One of the two most recent levels was drawn before the first dose."}
    placed = sorted([(a.time_h - doses[ia].start_h, a, ia), (b.time_h - doses[ib].start_h, b, ib)],
                    key=lambda x: x[0])
    (t1, first, i1), (t2, last, i2) = placed
    if ia == ib:
        dose = doses[ia]
        nxt = doses[ia + 1].start_h - dose.start_h if ia + 1 < len(doses) else tau_regimen
        tau, extra = nxt, {}
    else:
        lo, hi = min(ia, ib), max(ia, ib)
        run = doses[lo:hi + 1]
        if not all(_same_dose(run[0], d) for d in run[1:]):
            amounts = sorted({f"{d.amount_mg:g} mg over {d.duration_h:g} h" for d in run})
            return {"available": False,
                    "reason": ("The two most recent levels follow different doses, and the "
                               "doses between them were not all the same ("
                               + ", ".join(amounts) + "), so they cannot be placed in one "
                               "steady-state interval.")}
        gaps = [q.start_h - p.start_h for p, q in zip(run[:-1], run[1:])]
        if any(abs(g - tau_regimen) > 0.25 * tau_regimen for g in gaps):
            return {"available": False,
                    "reason": (f"The two most recent levels follow different doses, but those "
                               f"doses were not given every {tau_regimen:g} h (gaps of "
                               + ", ".join(f"{g:.1f}" for g in gaps)
                               + " h), so they cannot be placed in one interval.")}
        dose, tau = doses[ib], tau_regimen
        extra = {"extrapolated": True,
                 "doses_used": [{"time_h": doses[i].start_h, "amount_mg": doses[i].amount_mg,
                                 "infusion_h": doses[i].duration_h} for i in (i1, i2)]}
    if t1 < dose.duration_h:
        return {"available": False,
                "reason": ("Of the two most recent levels, the earlier one in the interval was "
                           f"drawn {t1:.1f} h after its dose started, before the "
                           f"{dose.duration_h:g} h infusion ended; the method needs both "
                           "levels after the infusion.")}
    if t2 - t1 < 0.5:
        return {"available": False,
                "reason": "The two most recent levels fall at almost the same time after "
                          "their doses, so they cannot give an elimination rate."}
    if last.conc_mg_l >= first.conc_mg_l:
        return {"available": False,
                "reason": ("Of the two most recent levels, the later one in the interval "
                           f"({last.conc_mg_l:g} mg/L at {t2:.1f} h) is not lower than the "
                           f"earlier one ({first.conc_mg_l:g} mg/L at {t1:.1f} h).")}
    return _pair_result(dose, tau, first, t1, last, t2, extra)


def two_level_from_history(doses: Sequence, levels: Sequence,
                           interval_h: Optional[float] = None) -> Dict[str, object]:
    """Two-level (Sawchuk-Zaske) AUC from a dosing history.

    Uses the two most recent levels. If they were drawn after different doses
    (a trough before dose N, a peak after it), they are placed in one interval
    when every dose between them is identical and given at the regimen's
    interval (steady state). If the two most recent levels cannot be used, falls
    back to the most recent dosing interval holding a usable pair, and says why.

    Returns a dict with ``available`` and, when False, a ``reason`` written for
    a clinician rather than a programmer. ``extrapolated`` marks a pair taken
    from two different doses.
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
    default_tau = interval_h if interval_h else (sorted(gaps)[len(gaps) // 2] if gaps else 24.0)

    recent = _most_recent_pair(doses, levels, default_tau)
    if recent["available"]:
        return recent

    # Fallback: the most recent single interval that holds a usable pair.
    for i in range(len(doses) - 1, -1, -1):
        dose = doses[i]
        tau = (starts[i + 1] - starts[i]) if i + 1 < len(doses) else default_tau
        window = [l for l in levels if dose.start_h <= l.time_h < dose.start_h + tau]
        if len(window) < 2:
            continue
        first, last = window[0], window[-1]
        t1 = first.time_h - dose.start_h
        t2 = last.time_h - dose.start_h
        if t1 < dose.duration_h or last.conc_mg_l >= first.conc_mg_l:
            continue
        out = _pair_result(dose, tau, first, t1, last, t2)
        out["note"] = ("The two most recent levels could not be used ("
                       + recent["reason"].rstrip(".") + "); this uses an earlier "
                       "pair from one dosing interval.")
        return out
    return recent
