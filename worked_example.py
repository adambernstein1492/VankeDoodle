"""Worked example: a 7-year-old boy with augmented renal clearance.

Runs the full VankeDoodle workflow end to end and prints each step:

  1. the patient
  2. the conventional starting regimen (20 mg/kg q8h)
  3. what each population model predicts that regimen will achieve, before any
     level is drawn
  4. a level comes back
  5. each model refits to it
  6. each model's dose recommendation
  7. the standard-of-care two-level comparison, and which method got closest

THE LEVELS IN THIS SCRIPT ARE INVENTED. They are generated from one defined
"true" individual so that the three values are mutually consistent and the
reference calculation has something coherent to work on. No patient data is
involved and nothing here is clinical evidence.

Run:  python worked_example.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vankedoodle as vd

pd.set_option("display.width", 200)

TARGET_LOW, TARGET_HIGH, TARGET = 400.0, 600.0, 500.0


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1. patient
rule("STEP 1  The patient")

patient = vd.Patient(
    weight_kg=23.0,          # 50th percentile for a 7-year-old boy
    age_years=7.0,
    height_cm=122.0,
    scr_mg_dl=0.33,          # low-normal creatinine -> augmented renal clearance
    sex="M",
    patient_id="worked-example",
)
print(patient.describe())
print(f"  bedside Schwartz CLcr   {patient.crcl:6.1f} mL/min/1.73 m2   (used by smit2021)")
print(f"  original Schwartz CLcr  {vd.schwartz_1976(122.0, 0.33, 7.0, 'M'):6.1f} "
      f"mL/min/1.73 m2   (used by lamarre2000)")
print(f"  serum creatinine        {patient.scr_umol_l:6.1f} umol/L            (used by xu2026)")
print(f"  postmenstrual age       {patient.pma_weeks:6.1f} weeks             "
      f"(used by colin2019, frymoyer2014)")
print("\nRenal function is above the usual range for age but not implausible -- the"
      "\nkind of child who is routinely under-dosed on a fixed mg/kg regimen.")


# ------------------------------------------------------- 2. starting regimen
rule("STEP 2  The conventional starting regimen")

mg_per_kg, interval_h, infusion_h = 20.0, 8.0, 1.0
dose_mg = vd.round_dose(mg_per_kg * patient.weight_kg)
print(f"{mg_per_kg:g} mg/kg = {mg_per_kg * patient.weight_kg:.0f} mg  ->  "
      f"{dose_mg:.0f} mg IV every {interval_h:g} h over {infusion_h:g} h")
print(f"  total daily dose {dose_mg * 24 / interval_h:.0f} mg "
      f"({dose_mg * 24 / interval_h / patient.weight_kg:.0f} mg/kg/day)")

regimen = vd.DosingHistory.repeating(dose_mg, interval_h, n_doses=8,
                                     infusion_h=infusion_h, start=0.0)


# ------------------------------------------------------------- 3. a priori
rule("STEP 3  What each model expects from that regimen, before any level")

def exposure_table(levels=()):
    rows = []
    for name in vd.MODELS:
        model = vd.get_model(name)
        try:
            fit = vd.fit(patient, regimen.doses, levels, model=name)
        except ValueError as exc:
            rows.append({"model": name, "note": str(exc)[:60]})
            continue
        exp = vd.predict_exposure(fit, dose_mg, interval_h, infusion_h,
                                  target_low=TARGET_LOW, target_high=TARGET_HIGH)
        out_of_pop = any("extrapolation" in w for w in fit.warnings)
        rows.append({
            "model": name + (" *" if out_of_pop else ""),
            "CL (L/h)": fit.parameters["CL"],
            "AUC24": exp.auc24,
            "90% CI": f"{exp.auc24_ci[0]:.0f}-{exp.auc24_ci[1]:.0f}",
            "P(target)": exp.prob_in_target,
            "peak d3": exp.cmax,
            "trough d3": exp.ctrough,
        })
    return pd.DataFrame(rows)


apriori = exposure_table()
print(apriori.round(2).to_string(index=False))
print("\n* marks a model used outside the population it was built in.")
print("  smit2021 is flagged because this child's CLcr exceeds the 120 mL/min/1.73 m2")
print("  cap built into that model: it structurally cannot represent augmented renal")
print(f"  clearance, so its clearance stops at {vd.SMIT2021.typical(patient)['CL']:.2f} L/h "
      f"where the uncapped covariate\n  relationship would give "
      f"{2.12 * (23.0/22.1)**0.745 * (patient.crcl/100):.2f} L/h. That is why it is the most "
      "optimistic model here.")
print("  frymoyer2014 is flagged because it is a neonatal model.")
inpop = apriori[~apriori["model"].str.endswith("*")]
print(f"\nAmong the {len(inpop)} applicable models the predicted AUC24 spans "
      f"{inpop['AUC24'].min():.0f}-{inpop['AUC24'].max():.0f} mg*h/L, and none gives "
      f"\nbetter than a {inpop['P(target)'].max()*100:.0f}% chance of landing in "
      f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}. That spread is the prior talking, not"
      "\nthe patient: no measurement has been made yet.")


# -------------------------------------------------------------- 4. a level
rule("STEP 4  A level comes back")

# The levels are generated under colin2019 -- an applicable model, deliberately
# not the default -- with clearance set well above its typical value. Whichever
# model generates the data is structurally favoured when the estimates are later
# compared, so this choice is stated rather than buried.
GENERATING_MODEL = "colin2019"
TRUE_CL = 3.6          # L/h -- the invented patient's real clearance
_eta_cl = float(np.log(TRUE_CL / vd.get_model(GENERATING_MODEL).typical(patient)["CL"]))
_sim = vd.simulate_levels(patient, regimen.doses, [23.5, 26.0, 31.5],
                          model=GENERATING_MODEL, eta=[_eta_cl, 0.0, 0.0],
                          residual_error=True, seed=11)
measured = {t: round(float(c), 1) for t, c in zip(_sim["times"], _sim["observed"])}
true_auc24 = dose_mg * 24.0 / interval_h / _sim["parameters"]["CL"]

print(f"INVENTED levels, generated under {GENERATING_MODEL} from one defined "
      f"individual\n(true CL = {_sim['parameters']['CL']:.2f} L/h, against a typical "
      f"{vd.get_model(GENERATING_MODEL).typical(patient)['CL']:.2f} L/h for that model):")
for t, c in measured.items():
    print(f"  t = {t:5.1f} h after the first dose : {c:5.1f} mg/L")
print(f"\nThat individual's true steady-state AUC24 is {true_auc24:.0f} mg*h/L.")
print("The engine is not told any of this; it only sees the levels.")
print("\nThe Bayesian route needs one of them: the trough at 23.5 h, drawn before")
print("the fourth dose. Steady state is not assumed or required.")

trough = [vd.Level(23.5, measured[23.5])]


# --------------------------------------------------------------- 5. refit
rule("STEP 5  Each model refits to that single level")

posterior = exposure_table(levels=trough)
print(posterior.round(2).to_string(index=False))

merged = apriori.merge(posterior, on="model", suffixes=(" prior", " post"))
print("\nMovement in estimated clearance:")
for _, r in merged.iterrows():
    print(f"  {r['model']:14s} {r['CL (L/h) prior']:5.2f} -> {r['CL (L/h) post']:5.2f} L/h"
          f"   AUC24 {r['AUC24 prior']:5.0f} -> {r['AUC24 post']:5.0f}")
inpop_post = posterior[~posterior["model"].str.endswith("*")]
print(f"\nThe applicable models now span {inpop_post['AUC24'].min():.0f}-"
      f"{inpop_post['AUC24'].max():.0f} mg*h/L (true {true_auc24:.0f}); one measurement has "
      "pulled them\ntogether and every one of them now says the child is under-exposed.")


# --------------------------------------------------------- 6. new regimen
rule("STEP 6  What each model recommends next")

rows = []
for name in vd.MODELS:
    fit = vd.fit(patient, regimen.doses, trough, model=name)
    rec = vd.recommend(fit, target_auc24=TARGET,
                       target_low=TARGET_LOW, target_high=TARGET_HIGH)
    best = rec.best
    rows.append({
        "model": name + (" *" if any("extrapolation" in w for w in fit.warnings) else ""),
        "dose": f"{best.dose_mg:.0f} mg q{best.interval_h:g}h",
        "mg/kg/day": best.mg_per_kg_per_day,
        "AUC24": best.auc24,
        "P(target)": best.prob_in_target,
        "peak d3": best.cmax,
        "trough d3": best.ctrough,
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nEvery model moves to q6h: at 23 kg the engine's conventional interval is q6h"
      "\n(the weight bands follow the Smit Table III guideline), and the starting q8h"
      "\nwas the thing being questioned. To keep the child on q8h, pin the interval:")

rows = []
for name in vd.MODELS:
    fit = vd.fit(patient, regimen.doses, trough, model=name)
    best = vd.recommend(fit, target_auc24=TARGET, interval_h=interval_h,
                        target_low=TARGET_LOW, target_high=TARGET_HIGH).best
    rows.append({
        "model": name + (" *" if any("extrapolation" in w for w in fit.warnings) else ""),
        "dose": f"{best.dose_mg:.0f} mg q{best.interval_h:g}h",
        "mg/kg/day": best.mg_per_kg_per_day,
        "AUC24": best.auc24,
        "P(target)": best.prob_in_target,
        "peak d3": best.cmax,
        "trough d3": best.ctrough,
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nSame daily dose either way -- AUC24 depends on total daily dose and clearance,"
      "\nnot on how the day is divided. The interval changes the peaks and troughs.")


# ------------------------------------------------- 7. the current standard
rule("STEP 7  The standard-of-care comparison (Aim 1 in miniature)")

print("Current practice draws two levels at steady state and applies Sawchuk-Zaske.")
print(f"  peak   t = 26.0 h  ({measured[26.0]:.1f} mg/L)  = 2.0 h into the 4th interval")
print(f"  trough t = 31.5 h  ({measured[31.5]:.1f} mg/L)  = 7.5 h into the 4th interval")

sz = vd.sawchuk_zaske(dose_mg=dose_mg, interval_h=interval_h, infusion_h=infusion_h,
                      t1_h=2.0, c1_mg_l=measured[26.0],
                      t2_h=7.5, c2_mg_l=measured[31.5])
print(f"\n  Sawchuk-Zaske: ke = {sz.ke:.4f} /h, t1/2 = {sz.half_life_h:.2f} h, "
      f"CL = {sz.clearance_l_h:.2f} L/h")
print(f"  Sawchuk-Zaske AUC24 = {sz.auc24:.0f} mg*h/L")

bayes_1level = {}
for name in vd.MODELS:
    fit = vd.fit(patient, regimen.doses, trough, model=name)
    if any("extrapolation" in w for w in fit.warnings):
        continue
    bayes_1level[name] = dose_mg * 24.0 / interval_h / fit.parameters["CL"]

print(f"\n  reference (the invented truth) = {true_auc24:.0f} mg*h/L")
print(f"  {'two levels, Sawchuk-Zaske':32s} {sz.auc24:6.0f}  "
      f"({100*(sz.auc24-true_auc24)/true_auc24:+5.1f}%)")
for name, auc in bayes_1level.items():
    print(f"  {'one level, Bayesian: ' + name:32s} {auc:6.0f}  "
          f"({100*(auc-true_auc24)/true_auc24:+5.1f}%)")

stats = vd.mpe_mape([true_auc24] * len(bayes_1level), list(bayes_1level.values()))
print(f"\n  Across the {stats['n']} applicable models, one Bayesian level gives "
      f"MPE {stats['mpe']:+.1f}% and MAPE {stats['mape']:.1f}%,")
print(f"  against {100*(sz.auc24-true_auc24)/true_auc24:+.1f}% for two levels by the "
      "reference method -- on one invented\n  patient, which proves nothing. Run it over "
      "the real cohort and it is Aim 1.")
print(f"\n  Read the per-model errors with care: the levels were generated under "
      f"{GENERATING_MODEL},\n  so that model is structurally favoured here. This example "
      "shows the workflow;\n  it is not a ranking of the models.")
