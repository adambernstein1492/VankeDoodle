"""VankeDoodle - Bayesian vancomycin dosing for children.

A thin interface over the engine's JSON contract: every number shown here comes
from ``vankedoodle.evaluate`` and nothing is recomputed in the page.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

import vankedoodle as vd
from app_charts import auc_chart, concentration_chart

st.set_page_config(page_title="VankeDoodle", layout="wide")

CASE_FILE = "cases.jsonl"

# Measured levels start EMPTY. Pre-filling them is unsafe: example concentrations
# left behind after a user edits the dose produce a confident-looking fit of a
# patient who does not exist. The example case is loaded only on request.
EMPTY_LEVELS = pd.DataFrame({"time": pd.Series(dtype="float64"),
                             "conc_mg_l": pd.Series(dtype="float64")})
EXAMPLE_LEVELS = pd.DataFrame({"time": [23.5, 26.0, 31.5],
                               "conc_mg_l": [6.2, 30.2, 8.3]})


@st.cache_data(show_spinner=False)
def run_engine(request_json: str) -> dict:
    """Evaluate one case. Cached so editing an unrelated widget costs nothing."""
    return vd.evaluate(json.loads(request_json))


def _fmt(value, spec=".0f", dash="-"):
    return dash if value is None else format(value, spec)


# ======================================================================
# Sidebar: the patient, the target, the model
# ======================================================================
with st.sidebar:
    st.header("Patient")
    weight_kg = st.number_input("Weight (kg)", min_value=0.4, max_value=250.0,
                                value=23.0, step=0.1)
    age_years = st.number_input("Age (years)", min_value=0.0, max_value=25.0,
                                value=7.0, step=0.5)
    height_cm = st.number_input("Height (cm)", min_value=20.0, max_value=220.0,
                                value=122.0, step=0.5)
    sex = st.selectbox("Sex", ["unknown", "M", "F"], index=1,
                       help="Only used by the Lamarre 2000 model, through the "
                            "Schwartz coefficient.")
    scr_unit = st.radio("Creatinine units", ["mg/dL", "µmol/L"], horizontal=True)
    scr_value = st.number_input(f"Serum creatinine ({scr_unit})",
                                min_value=0.01,
                                value=0.33 if scr_unit == "mg/dL" else 29.2,
                                step=0.01 if scr_unit == "mg/dL" else 1.0,
                                format="%.2f")

    gest_age = pna_days = None
    if age_years < 1.0:
        st.caption("Neonatal case: postmenstrual age is needed by the neonatal models.")
        gest_age = st.number_input("Gestational age at birth (weeks)",
                                   min_value=22.0, max_value=44.0, value=38.0, step=0.5)
        pna_days = st.number_input("Postnatal age (days)", min_value=0.0,
                                   max_value=400.0, value=14.0, step=1.0)

    patient = {"weight_kg": weight_kg, "age_years": age_years, "height_cm": height_cm}
    if sex != "unknown":
        patient["sex"] = sex
    patient["scr_mg_dl" if scr_unit == "mg/dL" else "scr_umol_l"] = scr_value
    if gest_age is not None:
        patient["gestational_age_weeks"] = gest_age
        patient["postnatal_age_days"] = pna_days

    st.header("Target")
    target_auc = st.number_input("Target AUC24 (mg·h/L)", min_value=200.0,
                                 max_value=900.0, value=500.0, step=10.0)
    target_low, target_high = st.slider("Acceptable range", 200, 900, (400, 600), step=10)

    st.header("Model")
    compare_all = st.checkbox("Compare all models", value=False)
    model_names = sorted(vd.MODELS)
    model = st.selectbox("Population model", model_names,
                         index=model_names.index("smit2021"),
                         help="The prior used for the Bayesian fit.")
    st.caption(vd.get_model(model).population)

# ======================================================================
# Header
# ======================================================================
st.title("VankeDoodle")
st.caption("Bayesian (model-informed) vancomycin dosing for children. "
           "Exposure estimates only - not a clinical decision, and not validated "
           "against clinical data.")

# ======================================================================
# Inputs: doses and levels
# ======================================================================
left, right = st.columns(2)

with left:
    st.subheader("Doses given")
    time_mode = st.radio("Enter times as", ["Hours after the first dose", "Clock times"],
                         horizontal=True)
    use_clock = time_mode == "Clock times"
    entry = st.radio("Dose entry", ["Regular schedule", "Individual doses"],
                     horizontal=True)

    if use_clock:
        c1, c2 = st.columns(2)
        first_date = c1.date_input("First dose date", value=dt.date.today())
        first_time = c2.time_input("First dose time", value=dt.time(8, 0))
        start = dt.datetime.combine(first_date, first_time).strftime("%Y-%m-%d %H:%M")
    else:
        start = 0.0

    if entry == "Regular schedule":
        c1, c2 = st.columns(2)
        dose_mg = c1.number_input("Dose (mg)", min_value=1.0, value=450.0, step=25.0,
                                  key="dose_mg_input")
        interval_h = c2.number_input("Interval (h)", min_value=1.0, max_value=72.0,
                                     value=8.0, step=1.0)
        c3, c4 = st.columns(2)
        n_doses = c3.number_input("Doses given so far", min_value=1, max_value=60, value=5)
        infusion_h = c4.number_input("Infusion duration (h)", min_value=0.25,
                                     max_value=24.0, value=1.0, step=0.25)
        st.caption(f"{dose_mg * 24 / interval_h:,.0f} mg/day = "
                   f"{dose_mg * 24 / interval_h / weight_kg:.0f} mg/kg/day")
        dose_block = {"regimen": {"dose_mg": dose_mg, "interval_h": interval_h,
                                  "n_doses": int(n_doses), "infusion_h": infusion_h,
                                  "start": start}}
    else:
        default_doses = pd.DataFrame({"time": [0.0, 8.0, 16.0, 24.0, 32.0],
                                      "amount_mg": [450.0] * 5,
                                      "infusion_h": [1.0] * 5})
        edited = st.data_editor(default_doses, num_rows="dynamic",
                                width='stretch', key="dose_table",
                                column_config={
                                    "time": st.column_config.NumberColumn(
                                        "Time (h after first dose)", step=0.25),
                                    "amount_mg": st.column_config.NumberColumn(
                                        "Dose (mg)", step=25.0),
                                    "infusion_h": st.column_config.NumberColumn(
                                        "Infusion (h)", step=0.25)})
        st.caption("Enter every dose actually given - late and missed doses included.")
        dose_block = {"doses": [{"amount_mg": r["amount_mg"], "time": r["time"],
                                 "infusion_h": r["infusion_h"]}
                                for _, r in edited.dropna().iterrows()]}

with right:
    st.subheader("Levels drawn")
    if "level_rev" not in st.session_state:
        st.session_state.level_rev = 0
        st.session_state.levels_df = EMPTY_LEVELS
    if st.button("Load the example case (three levels on 450 mg q8h)"):
        st.session_state.levels_df = EXAMPLE_LEVELS.copy()
        st.session_state.level_rev += 1
    levels_df = st.data_editor(
        st.session_state.levels_df, num_rows="dynamic", width='stretch',
        key=f"level_table_{st.session_state.level_rev}",
        column_config={
            "time": st.column_config.NumberColumn(
                "Time (h after first dose)", step=0.25),
            "conc_mg_l": st.column_config.NumberColumn("Measured (mg/L)", step=0.1)})
    st.caption("One level is enough for the Bayesian fit, and it need not be at "
               "steady state. Two within one interval also enable the Sawchuk-Zaske "
               "calculation. Levels must belong to the doses entered on the left - "
               "if you change the dose, change these too.")

    st.subheader("Try a specific regimen")
    c1, c2 = st.columns(2)
    cand_dose = c1.number_input("Dose (mg)", min_value=0.0, value=0.0, step=25.0,
                                key="cand_dose",
                                help="Leave at 0 to skip. Otherwise the predicted "
                                     "exposure for this regimen is shown below.")
    cand_interval = c2.number_input("Interval (h)", min_value=1.0, max_value=72.0,
                                    value=8.0, step=1.0, key="cand_int")

levels = [{"time": r["time"], "conc_mg_l": r["conc_mg_l"]}
          for _, r in levels_df.dropna().iterrows() if r["conc_mg_l"] > 0]

options = {"model": "all" if compare_all else model,
           "target_auc24": target_auc, "target_low": float(target_low),
           "target_high": float(target_high), "series": True}
if cand_dose > 0:
    options["candidate_regimen"] = {"dose_mg": cand_dose, "interval_h": cand_interval,
                                    "infusion_h": 1.0}

request = {"case_id": st.text_input("Case label (optional)", value="",
                                    placeholder="e.g. study ID or initials") or None,
           "patient": patient, "levels": levels, "options": options}
request.update(dose_block)

# ======================================================================
# Results
# ======================================================================
st.divider()
response = run_engine(json.dumps(request, sort_keys=True, default=str))

if not response["ok"]:
    for message in response["errors"]:
        st.error(message)
    st.stop()

shown = model if not compare_all else next(
    (r["model"] for r in response["results"] if r.get("ok")), model)
entry = next((r for r in response["results"] if r["model"] == shown), None)
if entry is None or not entry.get("ok"):
    st.error(entry["error"] if entry else f"{shown} produced no result.")
    st.stop()

current, rec = entry["current"], entry["recommendation"]
ci = current.get("auc24_ci") or [None, None]

given = response["regimen_given"]
if response["levels"]:
    st.caption(
        f"Fitting {len(response['levels'])} measured level(s) against "
        f"{given['n_doses']} dose(s) of {given['last_dose_mg']:.0f} mg every "
        f"{given['observed_interval_h']:.0f} h "
        f"({given['total_daily_dose_mg']:.0f} mg/day).")
else:
    st.caption(f"No levels entered - this is the population prediction for "
               f"{given['last_dose_mg']:.0f} mg every {given['observed_interval_h']:.0f} h. "
               "Add a level to individualise it.")

st.subheader("Where the patient is now")
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC24, current regimen", _fmt(current["auc24"]),
          "in target" if current["in_target"] else "off target",
          delta_color="normal" if current["in_target"] else "inverse")
m2.metric("90% interval", f"{_fmt(ci[0])} - {_fmt(ci[1])}")
m3.metric("P(in target)", f"{current['prob_in_target'] * 100:.0f}%"
          if current["prob_in_target"] is not None else "-")
m4.metric("Clearance", f"{entry['parameters']['CL']:.2f} L/h",
          f"{entry['n_levels_used']} level(s) used", delta_color="off")

for message in entry["warnings"]:
    st.warning(message, icon=":material/warning:")

st.subheader("Recommended next regimen")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Dose", f"{rec['dose_mg']:.0f} mg")
r2.metric("Interval", f"every {rec['interval_h']:.0f} h")
r3.metric("Predicted AUC24", _fmt(rec["auc24"]),
          f"{rec['prob_in_target'] * 100:.0f}% in target"
          if rec["prob_in_target"] is not None else None, delta_color="off")
r4.metric("Daily dose", f"{rec['mg_per_kg_per_day']:.0f} mg/kg/day")
st.caption(f"Infuse over {rec['infusion_h']:g} h. Predicted day-{rec['evaluation_day']} "
           f"peak {rec['cmax']:.1f} mg/L, trough {rec['ctrough']:.1f} mg/L. "
           f"Estimated with {entry['model']}.")
for flag in rec["flags"]:
    st.info(flag, icon=":material/info:")

if entry.get("candidate"):
    cand = entry["candidate"]
    st.caption(
        f"Your regimen - {cand['dose_mg']:.0f} mg every {cand['interval_h']:.0f} h: "
        f"predicted AUC24 {cand['auc24']:.0f} "
        f"({'in' if cand['in_target'] else 'outside'} target), day-"
        f"{cand['evaluation_day']} peak {cand['cmax']:.1f}, "
        f"trough {cand['ctrough']:.1f} mg/L.")

st.subheader("Concentration over time")
st.altair_chart(concentration_chart(response, shown), width='stretch')

if compare_all:
    st.subheader("How the models compare")
    st.altair_chart(auc_chart(response), width='stretch')
    st.caption("* marks a model used outside the population it was built in. "
               "The dashed red line, when present, is the two-level Sawchuk-Zaske value.")
    table = pd.DataFrame([
        {"model": r["model"], "ok": r["ok"],
         "CL (L/h)": r["parameters"]["CL"] if r["ok"] else None,
         "AUC24 now": r["current"]["auc24"] if r["ok"] else None,
         "recommended": (f"{r['recommendation']['dose_mg']:.0f} mg q"
                         f"{r['recommendation']['interval_h']:.0f}h") if r["ok"] else None,
         "out of population": r.get("out_of_population"),
         "note": r.get("error") or ""}
        for r in response["results"]])
    st.dataframe(table, width='stretch', hide_index=True)

st.subheader("Two-level calculation (current standard of care)")
sz = response["sawchuk_zaske"]
if sz["available"]:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("AUC24, two-level", _fmt(sz["auc24"]))
    s2.metric("Clearance", f"{sz['clearance_l_h']:.2f} L/h")
    s3.metric("Half-life", f"{sz['half_life_h']:.2f} h")
    diff = response["comparison"]["percent_difference"].get(shown)
    s4.metric("Bayesian vs two-level", f"{diff:+.1f}%" if diff is not None else "-")
    st.caption(
        f"Using the {sz['levels_used'][0]['conc_mg_l']:.1f} mg/L level at "
        f"{sz['levels_used'][0]['hours_into_interval']:.1f} h and the "
        f"{sz['levels_used'][1]['conc_mg_l']:.1f} mg/L level at "
        f"{sz['levels_used'][1]['hours_into_interval']:.1f} h into a "
        f"{sz['interval_h']:.0f} h interval.")
else:
    st.info(sz["reason"], icon=":material/info:")

# ======================================================================
# Keep the case
# ======================================================================
st.divider()
st.subheader("Save this case")
k1, k2 = st.columns(2)
if k1.button("Append to " + CASE_FILE, width='stretch'):
    vd.append_case(CASE_FILE, vd.case_record(response, model=shown))
    st.success(f"Saved. {len(vd.load_cases(CASE_FILE))} case(s) on file.")
k2.download_button("Download the full result as JSON",
                   data=json.dumps(response, indent=2),
                   file_name=f"vankedoodle_{response.get('case_id') or 'case'}.json",
                   mime="application/json", width='stretch')
