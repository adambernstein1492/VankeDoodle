"""Command-line interface: ``python -m vankedoodle ...``

Two subcommands mirror the two things a clinician does:

  start   -- empiric regimen, no levels yet
  assess  -- fit the measured level(s) and recommend the next regimen
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .workflow import assess, start_dose
from .dosing import AUC_TARGET, AUC_TARGET_HIGH, AUC_TARGET_LOW, smit_nomogram
from .models import DEFAULT_MODEL, MODELS, describe_models, get_model
from .api import append_case, case_record, evaluate
from .core import Patient


def _patient_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weight", type=float, required=True, help="total body weight, kg")
    p.add_argument("--age", type=float, default=None, help="age, years")
    p.add_argument("--height", type=float, default=None, help="height, cm")
    p.add_argument("--scr", type=float, default=None, help="serum creatinine, mg/dL")
    p.add_argument("--crcl", type=float, default=None,
                   help="bedside Schwartz CLcr, mL/min/1.73 m2 (instead of height+scr)")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                   help=f"population model used as the prior (default {DEFAULT_MODEL})")
    p.add_argument("--gest-age", type=float, default=None,
                   help="gestational age, weeks (preterm infants)")
    p.add_argument("--pna-days", type=float, default=None,
                   help="postnatal age, days (use instead of --age for neonates)")
    p.add_argument("--target", type=float, default=AUC_TARGET,
                   help=f"target AUC24, mg*h/L (default {AUC_TARGET:g})")
    p.add_argument("--target-low", type=float, default=AUC_TARGET_LOW)
    p.add_argument("--target-high", type=float, default=AUC_TARGET_HIGH)
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")


def _build_patient(a) -> Patient:
    return Patient(weight_kg=a.weight, age_years=a.age, height_cm=a.height,
                   scr_mg_dl=a.scr, crcl=a.crcl,
                   gestational_age_weeks=getattr(a, "gest_age", None),
                   postnatal_age_days=getattr(a, "pna_days", None))


def _exposure_dict(e) -> dict:
    return {"dose_mg": e.dose_mg, "interval_h": e.interval_h, "infusion_h": e.infusion_h,
            "total_daily_dose_mg": e.total_daily_dose_mg,
            "mg_per_kg_per_day": round(e.mg_per_kg_per_day, 2),
            "auc24": round(e.auc24, 1), "auc24_ci": e.auc24_ci,
            "prob_in_target": e.prob_in_target,
            "cmax": round(e.cmax, 2), "ctrough": round(e.ctrough, 2),
            "flags": e.flags}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vankedoodle",
        description="Bayesian (model-informed) vancomycin dosing for children aged 1-18 y.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list the population models available as priors")

    ev = sub.add_parser(
        "evaluate",
        help="run one case given as JSON and emit the full JSON response")
    ev.add_argument("case", nargs="?", default="-",
                    help="path to a request JSON file, or '-' for stdin (default)")
    ev.add_argument("--record", default=None, metavar="PATH",
                    help="also append a flat case record to this JSON Lines file")
    ev.add_argument("--record-model", default=None,
                    help="which model's result to record (default: the first that ran)")
    ev.add_argument("--indent", type=int, default=2)

    s = sub.add_parser("start", help="empiric regimen before any level is drawn")
    _patient_args(s)
    s.add_argument("--nomogram", action="store_true",
                   help="also show the published Smit 2021 fixed-dose guideline")

    b = sub.add_parser("assess", help="fit measured level(s) and recommend the next dose")
    _patient_args(b)
    b.add_argument("--dose", type=float, required=True, help="dose given, mg")
    b.add_argument("--interval", type=float, required=True, help="dosing interval, h")
    b.add_argument("--n-doses", type=int, default=1, help="number of doses given so far")
    b.add_argument("--infusion", type=float, default=1.0, help="infusion duration, h")
    b.add_argument("--start", default="0",
                   help="first dose time: 'YYYY-MM-DD HH:MM' or hours (default 0)")
    b.add_argument("--level", action="append", default=[], metavar="TIME,CONC",
                   help="a measured level, e.g. '2026-01-01 23:00,9.4' or '15,9.4'; repeatable")

    a = ap.parse_args(argv)
    if a.cmd == "models":
        print(describe_models())
        return 0
    if a.cmd == "evaluate":
        text = sys.stdin.read() if a.case == "-" else open(a.case, encoding="utf8").read()
        response = evaluate(json.loads(text))
        print(json.dumps(response, indent=a.indent))
        if a.record:
            if not response.get("ok"):
                return 1
            append_case(a.record, case_record(response, model=a.record_model))
        return 0 if response.get("ok") else 1
    pt = _build_patient(a)

    if a.cmd == "start":
        rec = start_dose(pt, target_auc24=a.target, model=a.model)
        if a.json:
            out = {"patient": {"weight_kg": pt.weight_kg,
                               "crcl": round(pt.crcl, 1) if pt.crcl else None},
                   "model": a.model, "basis": rec.basis, "target_auc24": a.target,
                   "recommended": _exposure_dict(rec.best),
                   "alternatives": [_exposure_dict(e) for e in rec.options[1:]],
                   "warnings": rec.warnings}
            if a.nomogram:
                out["smit_nomogram"] = smit_nomogram(pt)
            print(json.dumps(out, indent=2))
        else:
            print(rec.summary())
            if a.nomogram:
                n = smit_nomogram(pt)
                print(f"\nPublished Smit 2021 guideline: {n['mg_per_kg_per_dose']:g} mg/kg "
                      f"every {n['interval_h']:g} h = {n['dose_mg']:.0f} mg"
                      + (f", first dose {n['first_dose_mg']:.0f} mg"
                         if n["loading_dose_applied"] else ""))
        return 0

    start = a.start
    try:
        start = float(start)
    except ValueError:
        pass
    levels = []
    for raw in a.level:
        t, c = raw.rsplit(",", 1)
        t = t.strip()
        try:
            t = float(t)
        except ValueError:
            pass
        levels.append((t, float(c)))

    res = assess(pt,
                 {"dose_mg": a.dose, "interval_h": a.interval, "n_doses": a.n_doses,
                  "infusion_h": a.infusion, "start": start},
                 levels=levels, target_auc24=a.target, model=a.model,
                 target_low=a.target_low, target_high=a.target_high)
    if a.json:
        r = res.fit.residuals() if res.fit.levels else {}
        print(json.dumps({
            "patient": {"weight_kg": pt.weight_kg,
                        "crcl": round(pt.crcl, 1) if pt.crcl else None},
            "model": res.fit.model.name,
            "n_levels": len(res.fit.levels),
            "individual_parameters": {k: round(v, 4) for k, v in res.parameters.items()},
            "population_parameters": {k: round(v, 4)
                                      for k, v in res.fit.population_parameters.items()},
            "auc24_current_regimen": round(res.auc24_now, 1) if res.auc24_now else None,
            "in_target": res.in_target,
            "observed": [round(float(x), 2) for x in r.get("observed", [])],
            "predicted": [round(float(x), 2) for x in r.get("predicted", [])],
            "recommended": _exposure_dict(res.recommendation.best),
            "alternatives": [_exposure_dict(e) for e in res.recommendation.options[1:]],
            "warnings": res.fit.warnings,
        }, indent=2))
    else:
        print(res.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
