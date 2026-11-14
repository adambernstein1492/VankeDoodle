"""VankeDoodle - Bayesian vancomycin dosing for children.

A thin interface over the engine's JSON contract: every number shown here comes
from ``vankedoodle.evaluate`` and nothing clinical is recomputed in the page.
The case itself (patient, model, doses, levels, settings) is entered in the
sidebar; the regimen to try and the display options sit beside the results they
change.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

import vankedoodle as vd
import app_charts as charts
import app_report

ASSETS = Path(__file__).resolve().parent / "assets"
EXAMPLE_CASE = ASSETS.parent / "example_case.txt"
LOGO = ASSETS / "vankedoodle_logo.svg"           # transparent: sits on the theme background
TAB_ICON = ASSETS / "vankedoodle_icon.png"

st.set_page_config(page_title="VankeDoodle", layout="wide",
                   page_icon=str(TAB_ICON) if TAB_ICON.exists() else None)

UMOL_PER_MG_DL = 88.42
CLOCK_FORMAT = "%Y-%m-%d %H:%M"
SCHEDULE, INDIVIDUAL = "Regular schedule", "Individual doses"
AUTO = "Automatic"
FORECAST_H = 48.0           # how far past the last dose the plot forecasts
OLDEST_BIRTHDATE = dt.date.today() - dt.timedelta(days=round(30 * 365.25))

# The fields start filled in with a plausible patient and regimen so the page
# shows something useful on first run. Measured levels start EMPTY: example
# concentrations left behind after a dose is edited would fit a patient who does
# not exist. The example case, levels included, loads only on request.
DEFAULTS: Dict[str, Any] = {
    "weight_kg": 23.0, "height_cm": 122.0, "sex": "M",
    "birthdate": dt.date.today() - dt.timedelta(days=round(7 * 365.25)),
    "scr_unit": "mg/dL", "scr_value": 0.33, "gest_age": 38.0,
    "model": "smit2021", "dosing_model": "smit2021", "compare_all": False,
    **{f"overlay_{m}": False for m in vd.MODELS},
    "dose_entry": SCHEDULE,
    "first_date": dt.date.today(), "first_time": dt.time(8, 0),
    "dose_mg": 450.0, "interval_h": 8.0, "n_doses": 5, "infusion_h": 1.0,
    "cand_dose": 0.0, "cand_interval": 8.0, "cand_infusion": 1.0,
    "show_bands": True, "show_trough": True,
    "case_label": "",
    "table_version": 0,
    "dose_rows": [{"time_h": 8.0 * i, "amount_mg": 450.0, "infusion_h": 1.0}
                  for i in range(5)],
    "level_rows": [],
}
# Clinical settings, kept apart so they can be restored in one click. Bump the
# version whenever a default changes, so an open browser tab picks it up.
SETTINGS: Dict[str, Any] = {
    "target_auc": 500.0, "target_range": (400, 600), "peak_threshold": 50.0,
    "trough_range": (8.0, 12.0), "dose_increment": AUTO,
}
SETTINGS_VERSION = 2

for _key, _value in {**DEFAULTS, **SETTINGS}.items():
    st.session_state.setdefault(_key, _value)
ss = st.session_state
if ss.get("settings_version") != SETTINGS_VERSION:
    ss.update(SETTINGS)
    ss.settings_version = SETTINGS_VERSION


def _restore_settings() -> None:
    ss.update(SETTINGS)


@st.cache_data(show_spinner=False)
def run_engine(request_json: str) -> dict:
    """Evaluate one case. Cached so editing an unrelated widget costs nothing."""
    return vd.evaluate(json.loads(request_json))


def _origin() -> dt.datetime:
    return dt.datetime.combine(ss.first_date, ss.first_time)


# ======================================================================
# Loading a saved case and picking a row of the regimen grid both write into
# widget state, so they must run before the widgets are drawn.
# ======================================================================
def _hours(value, origin: Optional[dt.datetime]) -> float:
    """Hours after the first dose, from a clock time or a number of hours."""
    return vd.to_hours(value, origin) if origin else float(value)


def apply_case(case: Mapping) -> None:
    """Populate every sidebar input from a request in the evaluate() format."""
    reg, doses, levels = case.get("regimen"), case.get("doses") or [], case.get("levels") or []
    stamps = ([reg.get("start")] if reg else [d.get("time") for d in doses]) + \
             [l.get("time") for l in levels]
    clock = [pd.Timestamp(s.replace("/", "-")).to_pydatetime()
             for s in stamps if isinstance(s, str)]
    origin = min(clock) if clock else None    # cases given in hours keep today's first dose
    if origin:
        ss.first_date, ss.first_time = origin.date(), origin.time()

    p = case.get("patient") or {}
    for key in ("weight_kg", "height_cm"):
        if p.get(key) is not None:
            ss[key] = float(p[key])
    born = p.get("birthdate") or p.get("dob") or p.get("date_of_birth")
    if born:
        ss.birthdate = pd.Timestamp(str(born).replace("/", "-")).date()
    elif p.get("postnatal_age_days") is not None or p.get("age_years") is not None:
        # an age without a birthdate: count back from the first dose (or today)
        days = (float(p["postnatal_age_days"]) if p.get("postnatal_age_days") is not None
                else float(p["age_years"]) * 365.25)
        ss.birthdate = (origin.date() if origin else dt.date.today()) - \
            dt.timedelta(days=round(days))
    ss.birthdate = min(max(ss.birthdate, OLDEST_BIRTHDATE), dt.date.today())
    ss.sex = str(p.get("sex") or "unknown").upper().replace("UNKNOWN", "unknown")
    if p.get("scr_umol_l") is not None:
        ss.scr_unit, ss.scr_value = "µmol/L", float(p["scr_umol_l"])
    elif p.get("scr_mg_dl") is not None:
        ss.scr_unit, ss.scr_value = "mg/dL", float(p["scr_mg_dl"])
    if p.get("gestational_age_weeks") is not None:
        ss.gest_age = float(p["gestational_age_weeks"])

    if reg:
        ss.dose_entry = SCHEDULE
        ss.dose_mg = float(reg["dose_mg"])
        ss.interval_h = float(reg["interval_h"])
        ss.n_doses = int(reg.get("n_doses", 1))
        ss.infusion_h = float(reg.get("infusion_h", 1.0))
    if doses:
        ss.dose_entry = INDIVIDUAL
        ss.dose_rows = [{"time_h": _hours(d.get("time", d.get("start")), origin),
                         "amount_mg": float(d.get("amount_mg", d.get("dose_mg"))),
                         "infusion_h": float(d.get("infusion_h", 1.0))} for d in doses]
    ss.level_rows = [{"time_h": _hours(l["time"], origin), "conc_mg_l": float(l["conc_mg_l"])}
                     for l in levels]

    o = case.get("options") or {}
    if o.get("model") in vd.MODELS:
        ss.model = ss.dosing_model = o["model"]
    ss.compare_all = o.get("model") == "all"
    for src, dst in (("target_auc24", "target_auc"), ("peak_threshold_mg_l", "peak_threshold")):
        if o.get(src) is not None:
            ss[dst] = float(o[src])
    if o.get("target_low") is not None and o.get("target_high") is not None:
        ss.target_range = (int(o["target_low"]), int(o["target_high"]))
    cand = o.get("candidate_regimen")
    ss.cand_dose = float(cand["dose_mg"]) if cand else 0.0
    if cand:
        ss.cand_interval = float(cand["interval_h"])
        ss.cand_infusion = float(cand.get("infusion_h") or 1.0)
    ss.case_label = case.get("case_id") or ""
    ss.table_version += 1          # fresh editors, so they show the loaded rows


def read_upload(name: str, data: bytes) -> dict:
    """A request from an uploaded patient file (.txt) or request (.json)."""
    text = data.decode("utf-8-sig", errors="replace")
    if name.lower().endswith(".json"):
        return json.loads(text)
    return vd.read_case_text(text)


def load_example() -> None:
    """Button callback: the example patient file, levels and all."""
    apply_case(vd.read_case_text(EXAMPLE_CASE.read_text(encoding="utf8")))
    ss.load_message = ("success", "Loaded the example case (three levels on 450 mg q8h).")


upload = ss.get("case_upload")
if upload is not None and ss.get("loaded_case") != (upload.name, upload.size):
    ss.loaded_case = (upload.name, upload.size)
    try:
        apply_case(read_upload(upload.name, upload.getvalue()))
        ss.load_message = ("success", f"Loaded {upload.name}.")
    except vd.CaseFileError as exc:
        ss.load_message = ("error", f"Could not read {upload.name}:\n"
                           + "\n".join(f"- {p}" for p in exc.problems))
    except (ValueError, KeyError, TypeError) as exc:
        ss.load_message = ("error", f"Could not read {upload.name}: {exc}")

grid = ss.get("regimen_grid")
picked = list(grid["selection"]["rows"]) if grid and grid.get("selection") else []
if picked != ss.get("last_grid_pick", []):
    ss.last_grid_pick = picked
    rows = ss.get("grid_rows") or []
    if picked and picked[0] < len(rows):
        row = rows[picked[0]]
        ss.cand_dose = float(row["dose_mg"])
        ss.cand_interval = float(row["interval_h"])
        ss.cand_infusion = float(row["infusion_h"])


def _convert_scr() -> None:
    """Keep the creatinine value meaningful when its unit is switched."""
    if ss.scr_unit == "µmol/L":
        ss.scr_value = round(ss.scr_value * UMOL_PER_MG_DL, 1)
    else:
        ss.scr_value = round(ss.scr_value / UMOL_PER_MG_DL, 2)


def _age_on() -> dt.date:
    """The date the age is taken on: the date of the first dose."""
    return ss.first_date


def patient_age() -> Optional[Dict[str, float]]:
    """Age from the birthdate, by the engine's own rule; None if the birthdate is
    after the date the age is taken on (the engine then reports the error)."""
    try:
        return vd.age_from_birthdate(ss.birthdate, _age_on())
    except ValueError:
        return None


def _age_text(age: Optional[Dict[str, float]]) -> str:
    if age is None:
        return "-"
    if age["age_years"] < 1.0:
        return f"{age['age_days']} days"
    if age["age_years"] < 2.0:
        return f"{age['age_years']:.1f} years ({age['age_days']} days)"
    return f"{age['age_years']:.1f} years"


def patient_summary() -> pd.DataFrame:
    """The patient as entered, one row per field, for the sidebar table.

    Gestational age only appears for patients under one year, the same rule
    that decides whether its input is shown.
    """
    sex = {"M": "Male", "F": "Female"}.get(ss.sex, "Not given")
    scr = (f"{ss.scr_value:.2f} mg/dL" if ss.scr_unit == "mg/dL"
           else f"{ss.scr_value:.1f} µmol/L")
    age = patient_age()
    rows = [("Birthdate", ss.birthdate.strftime("%Y-%m-%d")),
            ("Age", _age_text(age)),
            ("Weight", f"{ss.weight_kg:g} kg"),
            ("Height", f"{ss.height_cm:g} cm"),
            ("Sex", sex),
            ("Serum creatinine", scr)]
    if age is not None and age["age_years"] < 1.0:
        rows += [("Gestational age", f"{ss.gest_age:g} weeks")]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _commit_tables(shift_h: float = 0.0) -> None:
    """Make the last edited tables the base rows and redraw the editors.

    The editors keep edits relative to the frame they were drawn from, so any
    change to that frame (a new first-dose time, a loaded case) goes through here.
    """
    for live, base in (("dose_rows_live", "dose_rows"), ("level_rows_live", "level_rows")):
        if ss.get(live) is not None:
            ss[base] = [dict(r, time_h=r["time_h"] + shift_h) for r in ss[live]]
    ss.table_version += 1


def _new_origin() -> None:
    """Moving the first dose keeps the clock times already entered where they are."""
    old = ss.get("origin_used")
    _commit_tables((old - _origin()).total_seconds() / 3600.0 if old else 0.0)


def _frame_to_rows(frame: pd.DataFrame) -> List[Dict[str, float]]:
    """Edited table rows, with clock times stored as hours after the first dose."""
    rows = []
    for _, r in frame.dropna().iterrows():
        t = (pd.Timestamp(r["time"]) - pd.Timestamp(_origin())).total_seconds() / 3600.0
        rows.append({"time_h": t, **{c: float(r[c]) for c in frame.columns if c != "time"}})
    return rows


def _rows_to_frame(rows: List[Dict[str, float]], columns: List[str]) -> pd.DataFrame:
    """Stored rows as an editable table with clock times."""
    frame = pd.DataFrame(rows, columns=columns)
    # Typed explicitly: an empty table (no levels yet) would otherwise have object
    # columns, which the editor refuses to show as clock times and numbers.
    frame["time"] = pd.to_datetime(pd.Series(
        [_origin() + dt.timedelta(hours=float(h)) for h in frame["time_h"]],
        dtype="object"))
    numeric = [c for c in columns if c != "time_h"]
    frame[numeric] = frame[numeric].astype("float64")
    return frame[["time"] + numeric]


def _time_value(value) -> str:
    """A table cell as the engine expects it: a clock string."""
    return pd.Timestamp(value).strftime(CLOCK_FORMAT)


def _time_column(label: str):
    return st.column_config.DatetimeColumn(label, format="YYYY-MM-DD HH:mm", step=60)


def _follow_population_model() -> None:
    """The dosing table follows the population model unless another is picked."""
    ss.dosing_model = ss.model


def dose_summary(block: Mapping) -> pd.DataFrame:
    """The doses being sent to the engine, one row each, for the sidebar table."""
    if "regimen" in block:
        r = block["regimen"]
        start = pd.Timestamp(r["start"])
        rows = [(start + pd.Timedelta(hours=i * r["interval_h"]), r["dose_mg"], r["infusion_h"])
                for i in range(int(r["n_doses"]))]
    else:
        rows = sorted((pd.Timestamp(d["time"]), d["amount_mg"], d["infusion_h"])
                      for d in block["doses"])
    return pd.DataFrame([{"#": i + 1, "Given": t.strftime("%m/%d %H:%M"),
                          "Dose (mg)": f"{mg:g}", "Infusion (h)": f"{h:g}"}
                         for i, (t, mg, h) in enumerate(rows)],
                        columns=["#", "Given", "Dose (mg)", "Infusion (h)"])


def level_summary(levels: List[Mapping]) -> pd.DataFrame:
    """The levels being sent to the engine, in time order, for the sidebar table."""
    rows = sorted((pd.Timestamp(l["time"]), l["conc_mg_l"]) for l in levels)
    return pd.DataFrame([{"#": i + 1, "Drawn": t.strftime("%m/%d %H:%M"),
                          "Level (mg/L)": f"{c:g}"} for i, (t, c) in enumerate(rows)],
                        columns=["#", "Drawn", "Level (mg/L)"])


# ======================================================================
# Sidebar: every input
# ======================================================================
with st.sidebar:
    if LOGO.exists():
        st.image(LOGO.read_text(encoding="utf8"), width="stretch")
    else:
        st.title("VankeDoodle")

    with st.expander("Load or save a case", icon=":material/folder_open:"):
        st.file_uploader("Load a patient file", type=["txt", "json"], key="case_upload",
                         help="A patient file (.txt) like example_case.txt, which can be "
                              "edited by hand to add doses or levels, or a request in "
                              "JSON (.json).")
        st.button("Load the example case", icon=":material/science:", width="stretch",
                  on_click=load_example,
                  help="A made-up 7-year-old on 450 mg q8h with three levels.")
        if ss.get("load_message"):
            kind, text = ss.load_message
            (st.success if kind == "success" else st.error)(text)
        st.text_input("Case label (optional)", key="case_label",
                      placeholder="e.g. study ID (no names or MRNs)",
                      help="Printed on the report. Use a study ID only: "
                           "no names, initials, MRNs or other identifiers.")
        download_slot = st.empty()

    make_report = st.button("Generate Report", icon=":material/picture_as_pdf:",
                            width="stretch",
                            help="A one-page PDF of this case: patient, fit, levels, "
                                 "concentration plot, exposure and dosing options.")
    report_slot = st.empty()

    # -------------------------------------------------------------- patient
    st.header("Patient")
    with st.popover("Edit patient data", icon=":material/person_edit:", width="stretch"):
        c1, c2 = st.columns(2)
        c1.date_input("Birthdate", min_value=OLDEST_BIRTHDATE, max_value=dt.date.today(),
                      format="YYYY-MM-DD", key="birthdate",
                      help="Age is worked out from this on the date of the first dose.")
        c2.selectbox("Sex", ["unknown", "M", "F"], key="sex",
                     help="Only used by the Lamarre 2000 model, through the Schwartz "
                          "coefficient.")
        c1.number_input("Weight (kg)", min_value=0.4, max_value=250.0, step=0.1,
                        key="weight_kg")
        c2.number_input("Height (cm)", min_value=20.0, max_value=220.0, step=0.5,
                        key="height_cm")
        st.radio("Creatinine units", ["mg/dL", "µmol/L"], horizontal=True, key="scr_unit",
                 on_change=_convert_scr)
        st.number_input(f"Serum creatinine ({ss.scr_unit})", min_value=0.01,
                        step=0.01 if ss.scr_unit == "mg/dL" else 1.0, format="%.2f",
                        key="scr_value")
        age = patient_age()
        if age is not None and age["age_years"] < 1.0:
            st.number_input("Gestational age at birth (weeks)", min_value=22.0,
                            max_value=44.0, step=0.5, key="gest_age",
                            help="With the birthdate this gives postmenstrual age, "
                                 "which the neonatal models need.")
    st.dataframe(patient_summary(), hide_index=True, width="stretch",
                 column_config={"Field": st.column_config.TextColumn(""),
                                "Value": st.column_config.TextColumn("")})

    # -------------------------------------------------------------- model
    st.header("Model")
    model_names = sorted(vd.MODELS)
    st.selectbox("Population model", model_names, key="model",
                 help="The prior used for the Bayesian fit.",
                 on_change=_follow_population_model)
    st.caption(vd.get_model(ss.model).population)
    st.checkbox("Also compare all models", key="compare_all")
    st.divider()

    # -------------------------------------------------------------- doses
    st.header("Vancomycin Dosing Information")
    with st.popover("Edit dosing information", icon=":material/medication:",
                    width="stretch"):
        c1, c2 = st.columns(2)
        c1.date_input("First dose date", key="first_date", on_change=_new_origin)
        c2.time_input("First dose time", key="first_time", on_change=_new_origin)
        st.radio("Dose entry", [SCHEDULE, INDIVIDUAL], horizontal=True, key="dose_entry")

        if ss.dose_entry == SCHEDULE:
            c1, c2 = st.columns(2)
            c1.number_input("Dose (mg)", min_value=1.0, step=25.0, key="dose_mg")
            c2.number_input("Interval (h)", min_value=1.0, max_value=72.0, step=1.0,
                            key="interval_h")
            c1.number_input("Doses given", min_value=1, max_value=60, key="n_doses")
            c2.number_input("Infusion (h)", min_value=0.25, max_value=24.0, step=0.25,
                            key="infusion_h")
            st.caption(f"{ss.dose_mg * 24 / ss.interval_h:,.0f} mg/day = "
                       f"{ss.dose_mg * 24 / ss.interval_h / ss.weight_kg:.0f} mg/kg/day")
            dose_block = {"regimen": {
                "dose_mg": ss.dose_mg, "interval_h": ss.interval_h,
                "n_doses": int(ss.n_doses), "infusion_h": ss.infusion_h,
                "start": _origin().strftime(CLOCK_FORMAT)}}
        else:
            dose_frame = st.data_editor(
                _rows_to_frame(ss.dose_rows, ["time_h", "amount_mg", "infusion_h"]),
                num_rows="dynamic", hide_index=True, width="stretch",
                key=f"dose_table_{ss.table_version}",
                column_config={"time": _time_column("Time given"),
                               "amount_mg": st.column_config.NumberColumn("Dose (mg)",
                                                                          step=25.0),
                               "infusion_h": st.column_config.NumberColumn("Infusion (h)",
                                                                           step=0.25)})
            ss.dose_rows_live = _frame_to_rows(dose_frame)
            st.caption("List every dose actually given - late and missed doses included.")
            dose_block = {"doses": [{"time": _time_value(r["time"]),
                                     "amount_mg": float(r["amount_mg"]),
                                     "infusion_h": float(r["infusion_h"])}
                                    for _, r in dose_frame.dropna().iterrows()]}
    doses_table = dose_summary(dose_block)
    if len(doses_table):
        st.dataframe(doses_table, hide_index=True, width="stretch",
                     column_config={"#": st.column_config.TextColumn("#", width="small")})
    else:
        st.caption("No doses entered yet.")
    st.divider()

    # -------------------------------------------------------------- levels
    st.header("Vancomycin Blood Levels")
    with st.popover("Edit blood levels", icon=":material/bloodtype:", width="stretch"):
        level_frame = st.data_editor(
            _rows_to_frame(ss.level_rows, ["time_h", "conc_mg_l"]),
            num_rows="dynamic", hide_index=True, width="stretch",
            key=f"level_table_{ss.table_version}",
            column_config={"time": _time_column("Time drawn"),
                           "conc_mg_l": st.column_config.NumberColumn("Measured (mg/L)",
                                                                      step=0.1)})
        st.caption("One level is enough for the Bayesian fit, and it need not be at "
                   "steady state. Two within one interval also enable the two-level "
                   "calculation.")
    ss.level_rows_live = _frame_to_rows(level_frame)
    ss.origin_used = _origin()
    levels = [{"time": _time_value(r["time"]), "conc_mg_l": float(r["conc_mg_l"])}
              for _, r in level_frame.dropna().iterrows() if r["conc_mg_l"] > 0]
    if levels:
        st.dataframe(level_summary(levels), hide_index=True, width="stretch",
                     column_config={"#": st.column_config.TextColumn("#", width="small")})
    else:
        st.caption("No levels entered yet: the population prior alone is used.")

    # -------------------------------------------------------------- settings
    st.divider()
    with st.popover("Settings", icon=":material/settings:", width="stretch"):
        st.number_input("Target AUC24 (mg·h/L)", min_value=200.0, max_value=900.0,
                        step=10.0, key="target_auc")
        st.slider("Acceptable AUC24 range", 200, 900, step=10, key="target_range")
        st.number_input("Peak ceiling (mg/L)", min_value=10.0, max_value=150.0, step=5.0,
                        key="peak_threshold",
                        help="Recommendations avoid regimens predicted to peak above "
                             "this; the peak risk is the chance of exceeding it.")
        st.slider("Trough range (mg/L)", 0.0, 30.0, step=0.5, key="trough_range",
                  help="Shaded on the concentration plot for reference.")
        st.selectbox("Dose rounding (mg)", [AUTO, 5.0, 10.0, 25.0, 50.0],
                     key="dose_increment",
                     help="Automatic rounds to 5 mg below 100 mg, 25 mg below 1 g, "
                          "and 50 mg above.")
        st.button("Restore defaults", on_click=_restore_settings, width="stretch",
                  icon=":material/restart_alt:")
    st.caption(f"Target AUC24 {ss.target_auc:g} ({ss.target_range[0]}-{ss.target_range[1]}) "
               f"· peak ceiling {ss.peak_threshold:g} mg/L · trough "
               f"{ss.trough_range[0]:g}-{ss.trough_range[1]:g} mg/L · rounding "
               f"{ss.dose_increment if ss.dose_increment == AUTO else f'{ss.dose_increment:g} mg'}")

# ======================================================================
# Build the request
# ======================================================================
patient: Dict[str, Any] = {"weight_kg": ss.weight_kg, "height_cm": ss.height_cm,
                           "birthdate": ss.birthdate.isoformat(),
                           "age_on": _age_on().isoformat()}
if ss.sex != "unknown":
    patient["sex"] = ss.sex
patient["scr_mg_dl" if ss.scr_unit == "mg/dL" else "scr_umol_l"] = ss.scr_value
_age = patient_age()
if _age is not None and _age["age_years"] < 1.0:
    patient["gestational_age_weeks"] = ss.gest_age

options: Dict[str, Any] = {
    "model": ss.model, "target_auc24": ss.target_auc,
    "target_low": float(ss.target_range[0]), "target_high": float(ss.target_range[1]),
    "peak_threshold_mg_l": ss.peak_threshold, "series": True, "uncertainty": True,
    "continuous_infusion": True, "forecast_h": FORECAST_H}
if ss.dose_increment != AUTO:
    options["dose_increment"] = float(ss.dose_increment)
if ss.cand_dose > 0:
    options["candidate_regimen"] = {"dose_mg": ss.cand_dose, "interval_h": ss.cand_interval,
                                    "infusion_h": ss.cand_infusion}

request = {"case_id": ss.case_label or None, "patient": patient, "levels": levels,
           "options": options, **dose_block}
_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", ss.case_label).strip("_") or "patient"
with download_slot.container():
    try:
        patient_file = vd.write_case_text(request)
    except ValueError as exc:                         # e.g. a birthdate after the doses
        patient_file = None
        st.caption(f"The patient file can't be written yet: {exc}")
    if patient_file:
        st.download_button("Save patient file (.txt)", data=patient_file,
                           file_name=f"{_stem}.txt", mime="text/plain", width="stretch",
                           icon=":material/download:",
                           help="The patient, model, doses and levels as entered now, in "
                                "the same format the loader reads. Edit it in any text "
                                "editor to add doses or levels.")
    st.download_button("Save as JSON", data=json.dumps(request, indent=2, default=str),
                       file_name=f"{_stem}.json", mime="application/json",
                       width="stretch")

peak = ss.peak_threshold

# ======================================================================
# Results
# ======================================================================
st.title("Pediatric Vancomycin Model-Informed Precision Dosing")
st.caption("Bayesian (model-informed) vancomycin dosing for children. Exposure "
           "estimates only - not a clinical decision, and not validated against "
           "clinical data.")

with st.spinner("Fitting the model and simulating uncertainty..."):
    response = run_engine(json.dumps(request, sort_keys=True, default=str))

if not response["ok"]:
    for message in response["errors"]:
        st.error(message)
    st.stop()

entry = response["results"][0]
if not entry.get("ok"):
    st.error(entry.get("error") or f"{ss.model} produced no result.")
    st.stop()

target = response["target"]


def _variant(**changes) -> dict:
    """The same case evaluated another way. The test regimen only affects the
    forecast, so it is left out and does not force these to be redone."""
    opts = {k: v for k, v in options.items() if k != "candidate_regimen"}
    opts.update(changes)
    return {**request, "options": opts}


# Every model, fitted without uncertainty: for overlays and the comparison.
overlay_models = [m for m in model_names if m != ss.model and ss[f"overlay_{m}"]]
all_models = None
if overlay_models or ss.compare_all:
    with st.spinner("Fitting the other models..."):
        all_models = run_engine(json.dumps(_variant(
            model="all", series=True, uncertainty=False, continuous_infusion=False),
            sort_keys=True, default=str))
    if not all_models["ok"]:
        all_models = None

# The dosing model: the population model unless another is picked. It sets the
# table's doses and the forecast (with the test regimen), so it is evaluated with
# uncertainty, the test regimen and its own fitted curve.
dosing_entry = entry
if ss.dosing_model != ss.model:
    with st.spinner(f"Fitting {ss.dosing_model}..."):
        other = run_engine(json.dumps({**request, "options": {**options,
                                                               "model": ss.dosing_model}},
                                      sort_keys=True, default=str))
    if other["ok"] and other["results"][0].get("ok"):
        dosing_entry = other["results"][0]
forecast_exposure = (dosing_entry["candidate"] if dosing_entry.get("candidate")
                     else dosing_entry["recommendation"])

# Levels that cannot be reconciled with the doses (an estimate beyond 3 prior SD)
# make every number derived from the fit meaningless. The fit is still drawn, so
# the effect of the wrong entry is visible, but no dose is recommended or
# forecast from it.
RECONCILE = "cannot be reconciled"
bad_fit = bool(entry.get("implausible_fit"))
bad_dosing = bool(dosing_entry.get("implausible_fit"))
if bad_fit:
    for message in entry["warnings"]:
        if RECONCILE in message:
            st.error(message, icon=":material/error:")
    st.caption("The fitted curve and current AUC24 below are what the levels imply. "
               "They are shown so you can see the effect of the entry that is wrong, "
               "not as a description of this patient.")

# -------------------------------------------------------------- report
# Built only when asked for, from exactly what the page shows, and offered for
# download only while the inputs still match it.
report_key = json.dumps([request, ss.dosing_model, ss.show_trough, ss.trough_range],
                        sort_keys=True, default=str)
if make_report:
    with st.spinner("Building the report..."):
        ss.report = (report_key, app_report.build_report(
            response, entry, dosing_entry, forecast_exposure,
            charts.regimen_options(dosing_entry), case_id=ss.case_label or None,
            trough_band=tuple(ss.trough_range) if ss.show_trough else None))
if ss.get("report"):
    built_for, pdf = ss.report
    if built_for == report_key:
        report_slot.download_button(
            "Download report (PDF)", data=pdf, mime="application/pdf",
            file_name=f"{_stem}_report_{dt.datetime.now():%Y%m%d_%H%M}.pdf",
            icon=":material/download:", width="stretch")
    else:
        report_slot.caption("The inputs have changed since the report was made; "
                            "generate it again.")

# -------------------------------------------------------------- concentration plot
left, right, side = st.columns([0.24, 0.52, 0.24])
with left:
    st.markdown("**Pharmacokinetic parameters**")
    st.dataframe(charts.parameter_table(entry), hide_index=True, width="stretch",
                 column_config={"Population": st.column_config.NumberColumn(format="%.2f"),
                                "Individual": st.column_config.NumberColumn(format="%.2f")})
    st.markdown("**Parameter uncertainty**")
    st.dataframe(charts.uncertainty_table(entry), hide_index=True, width="stretch",
                 column_config={"Prior CV (%)": st.column_config.NumberColumn(format="%.1f"),
                                "Post-fit CV (%)": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"Estimated with {entry['label']}. Only parameters the source study gave "
               "between-subject variability are individualised.")
with side:
    # Drop the column to the top of the plotting area: the plot's legend sits
    # above it, so the plot itself starts lower than the column does.
    st.markdown("<div style='height:3.2rem'></div>", unsafe_allow_html=True)
    display_col, compare_col = st.columns(2)
    with display_col:
        st.markdown("**Display**")
        st.checkbox("95% bands", key="show_bands")
        st.checkbox("Trough range", key="show_trough")
    with compare_col:
        st.markdown("**Compare models**")
        for m in model_names:
            if m != ss.model:
                st.checkbox(m, key=f"overlay_{m}", help=vd.get_model(m).population)
    if all_models:
        for r in all_models["results"]:
            if r["model"] in overlay_models and not r["ok"]:
                st.caption(f"{r['model']}: no fit - {r.get('error') or 'not applicable'}")
    st.divider()
    st.markdown(charts.auc_grid_html(
        [("Current", entry["current"])]
        + ([] if bad_dosing else [("Forecasted", forecast_exposure)]), target),
                unsafe_allow_html=True)
with right:
    bands = ss.show_bands
    st.plotly_chart(charts.concentration_figure(
        response,
        layers={"prior_band": bands, "fitted_band": bands, "forecast_band": bands,
                "forecast": not bad_dosing},
        trough_band=tuple(ss.trough_range) if ss.show_trough else None,
        peak_line=peak, overlay=all_models, overlay_models=overlay_models,
        forecast_from=dosing_entry, model_order=model_names),
        width="stretch")
    st.caption("The forecast follows the dosing model and test regimen under *Alternate "
               "Dosing Recommendations* below - the recommended regimen until you enter "
               "another. Click a legend entry to hide a curve.")

for message in entry["warnings"]:
    if bad_fit and RECONCILE in message:
        continue                      # already shown above, as an error
    st.warning(message, icon=":material/warning:")

# -------------------------------------------------------------- alternatives
st.divider()
st.subheader("Alternate Dosing Recommendations")
if bad_dosing:
    st.error(f"No dose is recommended while the levels and the doses disagree under "
             f"{dosing_entry['model']}. Correct the entry in the sidebar, or clear the "
             f"levels to see the population-based starting dose.",
             icon=":material/error:")
grid_rows = charts.regimen_options(dosing_entry)
ss.grid_rows = [{"dose_mg": r["dose_mg"], "interval_h": r["interval_h"],
                 "infusion_h": r["infusion_h"]} for r in grid_rows]


def _regimen_grid() -> None:
    st.dataframe(charts.regimen_table(grid_rows, target), hide_index=True,
                 width="content", key="regimen_grid", on_select="rerun",
                 selection_mode="single-row")


def _dosing_model_input() -> None:
    st.selectbox("Dosing model", model_names, key="dosing_model",
                 help="The model whose fit sets the table's doses and the forecast. "
                      "It follows the population model unless you pick another.")


def _test_inputs() -> None:
    st.number_input("Test dose (mg)", min_value=0.0, step=25.0, key="cand_dose",
                    help="The regimen forecast on the plot. At 0 the recommended "
                         "regimen is forecast; selecting a row fills these in.")
    st.number_input("Test interval (h)", min_value=1.0, max_value=72.0, step=1.0,
                    key="cand_interval")
    st.number_input("Infusion time (h)", min_value=0.25, max_value=24.0, step=0.25,
                    key="cand_infusion",
                    help="An infusion as long as the interval is a continuous infusion.")


if bad_dosing:
    # Nothing to tabulate or forecast; keep the model choice so another can be tried.
    with st.container(width=220):
        _dosing_model_input()
else:
    # The table takes only the width it needs and sits centred, with the test
    # regimen beside it. Horizontal containers need a recent Streamlit; older ones
    # get a centred column layout instead.
    try:
        row = st.container(horizontal=True, horizontal_alignment="center", gap="large",
                           vertical_alignment="top")
    except TypeError:
        row = None
    if row is not None:
        with row:
            with st.container(width=180):
                _dosing_model_input()
            _regimen_grid()
            with st.container(width=180):
                _test_inputs()
    else:
        model_col, table_col, test_col = st.columns([0.17, 0.63, 0.2])
        with model_col:
            _dosing_model_input()
        with table_col:
            _regimen_grid()
        with test_col:
            _test_inputs()
    st.markdown(
        f"<div style='text-align:center; font-size:0.85em; opacity:0.7;'>"
        f"Optimal dose: the dose that puts the steady-state AUC24 closest to "
        f"{target['auc24']:g} mg·h/L, rounded to a practical amount. %PTA: probability of "
        f"an AUC24 between {target['low']:g} and {target['high']:g}. "
        f"Select a row to forecast it on the plot.</div>", unsafe_allow_html=True)
    if dosing_entry is not entry:
        st.info(f"The doses above and the forecast use **{ss.dosing_model}**; the fitted "
                f"curve and current AUC24 use **{ss.model}**.", icon=":material/info:")
    elif ss.dosing_model != ss.model:
        st.warning(f"{ss.dosing_model} produced no fit for this case, so the table uses "
                   f"{ss.model}.", icon=":material/warning:")
    for flag in (dosing_entry["candidate"]["flags"] if dosing_entry.get("candidate") else []):
        st.info(flag, icon=":material/info:")

# -------------------------------------------------------------- two-level method
st.divider()
st.subheader("Two-level calculation (current standard of care)")
sz = response["sawchuk_zaske"]
if sz["available"]:
    plot_col, table_col = st.columns([0.55, 0.45], gap="large",
                                     vertical_alignment="center")
    with plot_col:
        st.markdown("**One interval: one-compartment vs the Bayesian model**")
        fig = charts.compartment_figure(response)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        first, second = sz["levels_used"][0], sz["levels_used"][1]
        st.caption(
            f"Using the {first['conc_mg_l']:.1f} mg/L level at "
            f"{first['hours_into_interval']:.1f} h and the {second['conc_mg_l']:.1f} mg/L "
            f"level at {second['hours_into_interval']:.1f} h into a {sz['interval_h']:g} h "
            f"interval. Cmax {sz['cmax']:.1f}, Cmin {sz['cmin']:.1f} mg/L.")
    with table_col:
        st.dataframe(charts.two_level_table(response), hide_index=True, width="content")
        st.caption("Difference: the Bayesian value relative to the two-level one.")
else:
    st.info(sz["reason"], icon=":material/info:")

# -------------------------------------------------------------- model comparison
if ss.compare_all:
    st.divider()
    st.subheader("How the models compare")
    comparison = all_models
    if comparison:
        # Plot and table side by side, the table sized to its contents.
        plot_col, table_col = st.columns([0.62, 0.38], gap="large",
                                         vertical_alignment="center")
        with plot_col:
            st.plotly_chart(charts.model_comparison_figure(comparison, x_range=(200, 800)),
                            width="stretch")
        with table_col:
            st.dataframe(charts.comparison_table(comparison), hide_index=True,
                         width="content")
        st.caption("Bars are 90% intervals. * marks a model used outside the population "
                   "it was built in. The dashed red line, when present, is the two-level "
                   "value.")
