"""When should the single early vancomycin level be drawn?

The study protocol draws one extra level within 24 h of starting vancomycin, after
one or two doses, and asks whether the Bayesian estimate of steady-state AUC24
from that single early sample matches the two-level steady-state calculation.
This script asks where in those 24 hours the sample is most informative.

Two questions are kept separate, because they have different answers:

  A. INFORMATION CONTENT.  Virtual subjects are generated from a model and fitted
     with that same model.  Any error is then due to the residual error and to how
     much the sample at that time constrains clearance -- not to the prior being
     wrong.  This is the sampling-design question.

  B. ROBUSTNESS TO THE PRIOR.  Virtual subjects are generated from one model and
     fitted with a different one.  Error now includes structural and covariate
     disagreement between models.  This is the "what if we picked the wrong
     prior" question.

The virtual subjects are synthetic: covariates are drawn from 50th-percentile
growth approximations with log-normal spread, and random effects from each
model's own published variance-covariance. They are a test of the estimator,
not clinical data, and nothing here is evidence about real patients.

Run:  python sampling_time_analysis.py
Writes sampling_time_results.csv.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

import vankedoodle as vd

# ---------------------------------------------------------------- settings
N_SUBJECTS = 200
MG_PER_KG, INTERVAL_H, INFUSION_H = 15.0, 6.0, 1.0
SAMPLE_TIMES = [1.5, 2.5, 4.0, 5.75, 7.5, 9.0, 11.75, 14.0, 17.75, 23.75]
TROUGH_TIMES = {5.75, 11.75, 17.75, 23.75}     # immediately pre-dose on a q6h schedule
MODELS = ["smit2021", "xu2026", "le2013", "lamarre2000", "colin2019"]
GENERATING_MODEL = "colin2019"                 # for analysis B
POP_SEED = 20260313

# 50th-percentile weight and height by age (boys), used only to build plausible
# virtual covariates.
_AGES = np.arange(1, 19)
_WT50 = np.array([10.0, 12.5, 14.5, 16.5, 18.5, 21.0, 23.0, 25.5, 28.5,
                  32.0, 36.0, 41.0, 46.0, 51.0, 56.0, 60.0, 63.0, 65.0])
_HT50 = np.array([76.0, 87.0, 95.0, 102.0, 109.0, 115.0, 122.0, 128.0, 133.0,
                  139.0, 144.0, 150.0, 157.0, 163.0, 169.0, 173.0, 175.0, 176.0])


def build_population(n: int = N_SUBJECTS, seed: int = POP_SEED) -> list:
    """A virtual cohort of children 1-18 y with plausible joint covariates."""
    rng = np.random.default_rng(seed)
    ages = rng.uniform(1.0, 18.0, n)
    wt = np.interp(ages, _AGES, _WT50) * np.exp(rng.normal(0.0, 0.18, n))
    ht = np.interp(ages, _AGES, _HT50) * np.exp(rng.normal(0.0, 0.04, n))
    # serum creatinine around the typical-for-age value used by Colin 2019
    from vankedoodle.models import colin_scr_std_mg_dl
    pma_yr = ages + 40.0 / 52.1775
    scr = np.array([colin_scr_std_mg_dl(p) for p in pma_yr]) * np.exp(
        rng.normal(0.0, 0.25, n))
    sex = rng.choice(["M", "F"], n)
    return [vd.Patient(weight_kg=float(w), age_years=float(a), height_cm=float(h),
                       scr_mg_dl=float(c), sex=str(s))
            for a, w, h, c, s in zip(ages, wt, ht, scr, sex)]


def regimen_for(patient) -> tuple:
    dose = vd.round_dose(MG_PER_KG * patient.weight_kg)
    doses = vd.DosingHistory.repeating(dose, INTERVAL_H, n_doses=10,
                                       infusion_h=INFUSION_H, start=0.0).doses
    return dose, doses


def _subject_truths(gen_model: str, population: list, seed: int) -> list:
    """One true individual per subject, drawn from the generating model's variability."""
    gen = vd.get_model(gen_model)
    rng = np.random.default_rng(seed)
    out = []
    for patient in population:
        eta = rng.multivariate_normal(np.zeros(gen.n_eta), gen.omega)
        pars = gen.individual(patient, eta)
        out.append((eta, pars, int(rng.integers(1 << 31))))
    return out


def _prior_errors(fit_model: str, population: list, truths: list) -> np.ndarray:
    """Error with no level at all -- independent of sampling time, so computed once."""
    errs = []
    for patient, (_, pars, _) in zip(population, truths):
        dose, _doses = regimen_for(patient)
        tdd = dose * 24.0 / INTERVAL_H
        true_auc = tdd / pars["CL"]
        fit = vd.fit(patient, [], [], model=fit_model, posterior_covariance=False)
        errs.append(100.0 * (tdd / fit.parameters["CL"] - true_auc) / true_auc)
    return np.array(errs)


def _cell(gen_model: str, fit_model: str, t_sample: float, population: list,
          truths: list, prior_err: np.ndarray) -> dict:
    """One (generating model, fitting model, sampling time) cell."""
    errors = []
    for patient, (eta, pars, seed) in zip(population, truths):
        dose, doses = regimen_for(patient)
        tdd = dose * 24.0 / INTERVAL_H
        true_auc = tdd / pars["CL"]
        sim = vd.simulate_levels(patient, doses, [t_sample], model=gen_model,
                                 eta=eta, residual_error=True, seed=seed)
        level = [vd.Level(t_sample, float(sim["observed"][0]))]
        fit = vd.fit(patient, doses, level, model=fit_model,
                     posterior_covariance=False)
        errors.append(100.0 * (tdd / fit.parameters["CL"] - true_auc) / true_auc)

    errors = np.array(errors)
    return {
        "generating_model": gen_model,
        "fitting_model": fit_model,
        "well_specified": gen_model == fit_model,
        "sample_time_h": t_sample,
        "is_trough": t_sample in TROUGH_TIMES,
        "n": int(errors.size),
        "mpe": float(np.mean(errors)),
        "mape": float(np.mean(np.abs(errors))),
        "within_20pct": float(np.mean(np.abs(errors) < 20.0) * 100),
        "p05": float(np.percentile(errors, 5)),
        "p95": float(np.percentile(errors, 95)),
        "prior_mape": float(np.mean(np.abs(prior_err))),
        "prior_mpe": float(np.mean(prior_err)),
    }


def main() -> pd.DataFrame:
    population = build_population()
    pairs = [(m, m) for m in MODELS]                                   # A
    pairs += [(GENERATING_MODEL, m) for m in MODELS if m != GENERATING_MODEL]   # B

    rows, t0 = [], time.perf_counter()
    for k, (gen_model, fit_model) in enumerate(pairs, 1):
        truths = _subject_truths(gen_model, population,
                                 seed=abs(hash(gen_model)) % (2 ** 31))
        prior_err = _prior_errors(fit_model, population, truths)
        for t_sample in SAMPLE_TIMES:
            rows.append(_cell(gen_model, fit_model, t_sample, population,
                              truths, prior_err))
        print(f"[{k}/{len(pairs)}] generate {gen_model} -> fit {fit_model}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("sampling_time_results.csv", index=False)
    print(f"\n{len(df)} cells x {N_SUBJECTS} virtual subjects "
          f"in {time.perf_counter() - t0:.0f}s")
    return df


if __name__ == "__main__":
    main()
