"""A plain-text patient file that can be written and edited by hand.

The file holds everything specific to one patient: demographics, creatinine,
the population model to use, and every vancomycin dose and level, each on its
own line. ``read_case_text`` turns it into a request for
:func:`vankedoodle.evaluate`; ``write_case_text`` turns a request back into the
same format, so a file can be loaded, added to, and saved again.

Example::

    # VankeDoodle patient file
    Patient ID: example
    Birthdate: 03/02/2019
    Sex: M
    Weight (kg): 23
    Height (cm): 122
    Creatinine (mg/dL): 0.33
    Model: smit2021

    [Doses]
    # Date       Time   Dose (mg)  Infusion (h)
    03/02/2026   08:00  450        1
    03/02/2026   16:00  450        1

    [Levels]
    # Date       Time   Level (mg/L)
    03/03/2026   07:30  6.2

Reading is forgiving: keys are case-insensitive and their units optional,
values may be separated by spaces, tabs or commas, infusion time defaults to
one hour, and ISO dates (YYYY-MM-DD) are accepted as well as MM/DD/YYYY, so
files from vancydoodle load too. Every problem found is reported together,
with its line number.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .models import MODELS

DATE_FORMAT = "%m/%d/%Y"
TIME_FORMAT = "%H:%M"
DEFAULT_INFUSION_H = 1.0

__all__ = ["CaseFileError", "read_case_text", "write_case_text"]


class CaseFileError(ValueError):
    """A patient file that could not be read; the message lists every problem,
    each with its line number."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
_SECTIONS = {"doses": "doses", "dose": "doses", "levels": "levels", "level": "levels",
             "concentrations": "levels", "patient": "patient"}


def _parse_date(text: str) -> date:
    text = text.strip()
    for fmt in (DATE_FORMAT, "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"'{text}' is not a date (use MM/DD/YYYY)")


def _parse_time(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text.strip())
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"'{text}' is not a time (use 24-hour HH:MM)")
    return int(m.group(1)), int(m.group(2))


def _number(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{what} '{text}' is not a number") from None
    if value <= 0:
        raise ValueError(f"{what} must be greater than zero")
    return value


def _fields(line: str) -> List[str]:
    return [f for f in re.split(r"[,\t ]+", line.strip()) if f]


def _timed_row(line: str, n_values: int, names: Tuple[str, ...]) -> Tuple[str, List[float]]:
    """Date, time and up to ``n_values`` numbers from one line."""
    parts = _fields(line)
    if len(parts) < 3:
        raise ValueError(f"expected a date, a time and {names[0]}")
    d = _parse_date(parts[0])
    hh, mm = _parse_time(parts[1])
    values = parts[2:]
    if len(values) > n_values:
        raise ValueError(f"too many values; expected {' and '.join(names)}")
    stamp = datetime(d.year, d.month, d.day, hh, mm).strftime("%Y-%m-%d %H:%M")
    return stamp, [_number(v, n) for v, n in zip(values, names)]


def _model_name(text: str) -> str:
    wanted = text.strip().lower()
    for name in MODELS:
        if name.lower() == wanted:
            return name
    raise ValueError(f"unknown model '{text.strip()}' (choose from {', '.join(sorted(MODELS))})")


def read_case_text(text: str) -> Dict[str, Any]:
    """Read a patient file into a request for :func:`vankedoodle.evaluate`.

    Raises :class:`CaseFileError` listing every problem, each with its line number.
    """
    patient: Dict[str, Any] = {}
    options: Dict[str, Any] = {}
    doses: List[Dict[str, Any]] = []
    levels: List[Dict[str, Any]] = []
    case_id: Optional[str] = None
    problems: List[str] = []
    attempted = set()          # fields given, even if their value was rejected
    dose_lines = 0
    section = "patient"

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        header = re.fullmatch(r"\[\s*([A-Za-z ]+?)\s*\]", line)
        if header:
            key = header.group(1).lower()
            if key not in _SECTIONS:
                problems.append(f"Line {number}: unknown section [{header.group(1)}] "
                                "(use [Doses] or [Levels])")
                section = "unknown"
            else:
                section = _SECTIONS[key]
            continue
        try:
            if section == "patient":
                if ":" not in line:
                    raise ValueError("expected 'Name: value'")
                key, value = (s.strip() for s in line.split(":", 1))
                unit = (re.search(r"\(([^)]*)\)", key) or [None, ""])[1].lower()
                key = re.sub(r"\([^)]*\)", "", key).strip().lower()
                if not value:
                    continue                                  # left blank on purpose
                attempted.add({"birth date": "birthdate", "date of birth": "birthdate",
                               "dob": "birthdate", "scr": "creatinine",
                               "serum creatinine": "creatinine"}.get(key, key))
                if key in ("patient id", "patient", "id", "case id", "label", "name"):
                    case_id = value
                elif key in ("birthdate", "birth date", "date of birth", "dob"):
                    patient["birthdate"] = _parse_date(value).isoformat()
                elif key == "weight":
                    if unit not in ("", "kg"):
                        raise ValueError(f"weight must be in kg, not {unit}")
                    patient["weight_kg"] = _number(value, "Weight")
                elif key == "height":
                    if unit not in ("", "cm"):
                        raise ValueError(f"height must be in cm, not {unit}")
                    patient["height_cm"] = _number(value, "Height")
                elif key == "sex":
                    sex = value.strip().upper()[:1]
                    if sex not in ("M", "F"):
                        raise ValueError(f"sex '{value}' should be M or F")
                    patient["sex"] = sex
                elif key in ("creatinine", "scr", "serum creatinine"):
                    v = _number(value, "Creatinine")
                    if "umol" in unit or "µmol" in unit or "μmol" in unit:
                        patient["scr_umol_l"] = v
                    else:
                        patient["scr_mg_dl"] = v
                elif key in ("gestational age", "ga"):
                    patient["gestational_age_weeks"] = _number(value, "Gestational age")
                elif key == "model":
                    options["model"] = _model_name(value)
                else:
                    raise ValueError(f"unknown field '{key}'")
            elif section == "doses":
                dose_lines += 1
                stamp, vals = _timed_row(line, 2, ("a dose (mg)", "an infusion time (h)"))
                if not vals:
                    raise ValueError("missing the dose (mg)")
                doses.append({"time": stamp, "amount_mg": vals[0],
                              "infusion_h": vals[1] if len(vals) > 1 else DEFAULT_INFUSION_H})
            elif section == "levels":
                stamp, vals = _timed_row(line, 1, ("a level (mg/L)",))
                if not vals:
                    raise ValueError("missing the level (mg/L)")
                levels.append({"time": stamp, "conc_mg_l": vals[0]})
        except ValueError as exc:
            problems.append(f"Line {number}: {exc}")

    for key, field, label in (("birthdate", "birthdate", "Birthdate"),
                              ("weight", "weight_kg", "Weight (kg)"),
                              ("height", "height_cm", "Height (cm)")):
        if field not in patient and key not in attempted:
            problems.append(f"Missing '{label}'")
    if ("scr_mg_dl" not in patient and "scr_umol_l" not in patient
            and "creatinine" not in attempted):
        problems.append("Missing 'Creatinine (mg/dL)'")
    if not doses and not dose_lines:
        problems.append("No doses: list them under [Doses]")
    if problems:
        raise CaseFileError(problems)

    request: Dict[str, Any] = {"patient": patient,
                               "doses": sorted(doses, key=lambda d: d["time"]),
                               "levels": sorted(levels, key=lambda l: l["time"]),
                               "options": options}
    if case_id:
        request["case_id"] = case_id
    return request


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
def _stamp(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(str(value).strip().replace("T", " ")[:16], "%Y-%m-%d %H:%M")


def _num(x: float) -> str:
    return f"{float(x):g}"


def _expand_regimen(reg: Mapping) -> List[Dict[str, Any]]:
    from datetime import timedelta
    if not isinstance(reg.get("start"), str):
        raise ValueError("A patient file needs clock times; the regimen has no start date.")
    start = _stamp(reg["start"])
    return [{"time": (start + timedelta(hours=i * float(reg["interval_h"]))).strftime(
                 "%Y-%m-%d %H:%M"),
             "amount_mg": reg["dose_mg"], "infusion_h": reg.get("infusion_h", 1.0)}
            for i in range(int(reg.get("n_doses", 1)))]


def write_case_text(request: Mapping) -> str:
    """Write a request (as built for :func:`vankedoodle.evaluate`) as a patient file.

    The request must use clock times and carry a birthdate. A regular schedule
    is written out dose by dose, so the file can be edited one dose at a time.
    """
    p = request.get("patient") or {}
    if not p.get("birthdate"):
        raise ValueError("A patient file needs the birthdate.")
    doses = list(request.get("doses") or [])
    if request.get("regimen"):
        doses = _expand_regimen(request["regimen"])
    for d in doses + list(request.get("levels") or []):
        if not isinstance(d.get("time"), str):
            raise ValueError("A patient file needs clock times for every dose and level.")

    born = date.fromisoformat(str(p["birthdate"])[:10])
    lines = ["# VankeDoodle patient file",
             "# Lines starting with # are notes and are ignored.",
             "# Dates are MM/DD/YYYY and times are 24-hour HH:MM.",
             ""]
    if request.get("case_id"):
        lines.append(f"Patient ID: {request['case_id']}")
    lines.append(f"Birthdate: {born.strftime(DATE_FORMAT)}")
    if p.get("sex"):
        lines.append(f"Sex: {p['sex']}")
    lines.append(f"Weight (kg): {_num(p['weight_kg'])}")
    lines.append(f"Height (cm): {_num(p['height_cm'])}")
    if p.get("scr_umol_l") is not None:
        lines.append(f"Creatinine (umol/L): {_num(p['scr_umol_l'])}")
    else:
        lines.append(f"Creatinine (mg/dL): {_num(p['scr_mg_dl'])}")
    if p.get("gestational_age_weeks") is not None:
        lines.append(f"Gestational age (weeks): {_num(p['gestational_age_weeks'])}")
    model = (request.get("options") or {}).get("model")
    if model and model != "all":
        lines.append(f"Model: {model}")

    lines += ["", "[Doses]", "# Date       Time   Dose (mg)  Infusion (h)"]
    for d in sorted(doses, key=lambda d: _stamp(d["time"])):
        t = _stamp(d["time"])
        lines.append(f"{t.strftime(DATE_FORMAT)}   {t.strftime(TIME_FORMAT)}  "
                     f"{_num(d.get('amount_mg', d.get('dose_mg'))):<10} "
                     f"{_num(d.get('infusion_h', DEFAULT_INFUSION_H))}")

    lines += ["", "[Levels]", "# Date       Time   Level (mg/L)"]
    for l in sorted(request.get("levels") or [], key=lambda l: _stamp(l["time"])):
        t = _stamp(l["time"])
        lines.append(f"{t.strftime(DATE_FORMAT)}   {t.strftime(TIME_FORMAT)}  "
                     f"{_num(l['conc_mg_l'])}")
    return "\n".join(lines) + "\n"
