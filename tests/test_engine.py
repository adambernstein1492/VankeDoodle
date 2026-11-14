"""Tests for the VankeDoodle engine.

Correctness is checked three ways:
  * against closed-form pharmacokinetic results (mass balance, one-compartment limit);
  * against the published source model (typical parameter values at the reference
    covariates, and the paper's own Table III dosing guideline reproducing its
    simulated day-3 exposures);
  * by simulation-based recovery of known individual parameters.

The simulated subjects here are virtual patients used to test the estimator.
They are not clinical data and say nothing about performance in real patients.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import vankedoodle as vd
from vankedoodle.core import Dose
from vankedoodle.dosing import smit_nomogram
from vankedoodle.pk import TwoCompartmentModel, model_from_params


# ---------------------------------------------------------------- covariates
def test_bedside_schwartz():
    assert vd.bedside_schwartz(100.0, 0.5) == pytest.approx(0.413 * 100 / 0.5)


def test_typical_parameters_at_reference_covariates():
    """At TBW 22.1 kg and CLcr 100 the model must return the published estimates."""
    pt = vd.Patient(weight_kg=22.1, crcl=100.0, age_years=6)
    p = vd.SMIT2021.typical(pt)
    assert p["CL"] == pytest.approx(2.12, rel=1e-9)
    assert p["V1"] == pytest.approx(8.90, rel=1e-9)
    assert p["Q"] == pytest.approx(1.55, rel=1e-9)
    assert p["V2"] == pytest.approx(12.3, rel=1e-9)


def test_clearance_scaling():
    base = vd.Patient(weight_kg=22.1, crcl=100.0)
    heavy = vd.Patient(weight_kg=44.2, crcl=100.0)
    renal = vd.Patient(weight_kg=22.1, crcl=50.0)
    assert vd.SMIT2021.typical(heavy)["CL"] / vd.SMIT2021.typical(base)["CL"] == \
        pytest.approx(2.0 ** 0.745)
    assert vd.SMIT2021.typical(renal)["CL"] == pytest.approx(0.5 * 2.12)
    # volumes scale linearly with weight
    assert vd.SMIT2021.typical(heavy)["V1"] == pytest.approx(2.0 * 8.90)


def test_crcl_capped_at_120():
    high = vd.Patient(weight_kg=22.1, crcl=200.0)
    cap = vd.Patient(weight_kg=22.1, crcl=120.0)
    assert vd.SMIT2021.typical(high)["CL"] == pytest.approx(vd.SMIT2021.typical(cap)["CL"])


def test_omega_matches_reported_cvs():
    om = vd.SMIT2021.omega
    assert math.sqrt(math.exp(om[0, 0]) - 1) == pytest.approx(0.287, abs=1e-9)
    assert math.sqrt(math.exp(om[1, 1]) - 1) == pytest.approx(1.10, abs=1e-9)
    assert om[0, 1] == pytest.approx(-0.085)
    assert np.all(np.linalg.eigvalsh(om) > 0)       # a valid covariance matrix


# ---------------------------------------------------------------- PK solver
def test_bolus_initial_concentration():
    pk = TwoCompartmentModel(CL=2.0, V1=10.0, Q=1.5, V2=12.0)
    c = pk.concentrations([Dose(amount_mg=500, start_h=0.0, duration_h=0.0)], [0.0])
    assert c[0] == pytest.approx(50.0)


def test_total_auc_equals_dose_over_clearance():
    """Mass balance: the AUC to infinity of a single dose is Dose/CL."""
    pk = TwoCompartmentModel(CL=2.0, V1=10.0, Q=1.5, V2=12.0)
    auc = pk.auc([Dose(amount_mg=500, start_h=0.0, duration_h=1.0)], 0.0, 2000.0)
    assert auc == pytest.approx(500.0 / 2.0, rel=1e-6)


def test_steady_state_auc_equals_tdd_over_cl():
    """Over one interval at steady state, AUC = dose/CL regardless of interval."""
    pk = TwoCompartmentModel(CL=2.5, V1=9.0, Q=1.5, V2=12.0)
    for tau in (6.0, 8.0, 12.0, 24.0):
        doses = [Dose(amount_mg=400, start_h=i * tau, duration_h=1.0) for i in range(300)]
        t0 = 250 * tau
        assert pk.auc(doses, t0, t0 + tau) == pytest.approx(400.0 / 2.5, rel=1e-5)


def test_one_compartment_limit_matches_closed_form():
    """With negligible distribution the solver must match the textbook infusion equation."""
    CL, V = 2.0, 10.0
    pk = TwoCompartmentModel(CL=CL, V1=V, Q=1e-9, V2=1e-6)
    ke = CL / V
    T, dose = 1.0, 500.0
    rate = dose / T
    for t in (0.25, 0.5, 1.0, 2.0, 6.0):
        got = pk.concentrations([Dose(amount_mg=dose, start_h=0.0, duration_h=T)], [t])[0]
        if t <= T:
            want = rate / CL * (1 - math.exp(-ke * t))
        else:
            want = rate / CL * (1 - math.exp(-ke * T)) * math.exp(-ke * (t - T))
        assert got == pytest.approx(want, rel=1e-6)


def test_irregular_and_missed_doses():
    """A late/missed dose must be handled by superposition, not by assuming a schedule."""
    pk = TwoCompartmentModel(CL=2.0, V1=10.0, Q=1.5, V2=12.0)
    on_time = [Dose(500, 0.0, 1.0), Dose(500, 8.0, 1.0), Dose(500, 16.0, 1.0)]
    delayed = [Dose(500, 0.0, 1.0), Dose(500, 12.0, 1.0), Dose(500, 20.0, 1.0)]
    assert pk.auc(on_time, 0, 1000) == pytest.approx(pk.auc(delayed, 0, 1000), rel=1e-6)
    # but the trough before the delayed dose must be lower
    assert pk.concentrations(delayed, [11.9])[0] < pk.concentrations(on_time, [11.9])[0]


def test_overlapping_infusions_superpose():
    pk = TwoCompartmentModel(CL=2.0, V1=10.0, Q=1.5, V2=12.0)
    a = pk.concentrations([Dose(500, 0.0, 2.0)], [1.0])[0]
    b = pk.concentrations([Dose(300, 0.5, 2.0)], [1.0])[0]
    both = pk.concentrations([Dose(500, 0.0, 2.0), Dose(300, 0.5, 2.0)], [1.0])[0]
    assert both == pytest.approx(a + b, rel=1e-9)


def test_half_lives_ordered():
    pk = TwoCompartmentModel(CL=2.0, V1=10.0, Q=1.5, V2=12.0)
    dist, term = pk.half_lives
    assert 0 < dist < term


# ---------------------------------------------------------------- estimation
def _patient():
    return vd.Patient(weight_kg=20.0, age_years=6.0, height_cm=115.0, scr_mg_dl=0.4)


def test_fit_without_levels_returns_prior():
    f = vd.fit(_patient(), doses=[], levels=[])
    assert np.allclose(f.eta, 0.0)
    assert f.parameters == f.population_parameters


def test_fit_recovers_known_individual_from_dense_noise_free_levels():
    pt = _patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 8).doses
    eta_true = np.array([0.35, -0.60])
    times = [5.0, 11.5, 23.0, 29.5, 41.0, 47.0]
    sim = vd.simulate_levels(pt, doses, times, eta=eta_true, residual_error=False)
    levels = [vd.Level(time_h=t, conc_mg_l=c) for t, c in zip(times, sim["observed"])]
    f = vd.fit(pt, doses, levels)
    assert f.parameters["CL"] == pytest.approx(sim["parameters"]["CL"], rel=0.02)
    assert f.eta[0] == pytest.approx(eta_true[0], abs=0.03)


def test_levels_move_estimate_in_the_right_direction():
    pt = _patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 6).doses
    prior_cl = vd.SMIT2021.typical(pt)["CL"]
    prior_trough = model_from_params(vd.SMIT2021.typical(pt)).concentrations(doses, [23.9])[0]
    low = vd.fit(pt, doses, [vd.Level(23.9, prior_trough * 0.5)])
    high = vd.fit(pt, doses, [vd.Level(23.9, prior_trough * 2.0)])
    assert low.parameters["CL"] > prior_cl      # lower level => faster clearance
    assert high.parameters["CL"] < prior_cl


def test_posterior_narrows_with_more_levels():
    pt = _patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 8).doses
    sim = vd.simulate_levels(pt, doses, [5.0, 11.5, 23.0, 29.5], eta=[0.2, -0.3],
                             residual_error=True, seed=7)
    lv = [vd.Level(t, c) for t, c in zip(sim["times"], sim["observed"])]
    one = vd.fit(pt, doses, lv[:1])
    four = vd.fit(pt, doses, lv)
    assert four.eta_covariance[0, 0] < one.eta_covariance[0, 0] < vd.SMIT2021.omega[0, 0]


def test_simulation_based_auc_recovery_is_unbiased():
    """Across virtual subjects, MAP AUC24 estimates from two levels track the truth."""
    rng = np.random.default_rng(20260101)
    errors = []
    for _ in range(40):
        wt = float(rng.uniform(8, 80))
        crcl = float(rng.uniform(40, 130))
        pt = vd.Patient(weight_kg=wt, crcl=crcl, age_years=10.0)
        dose = round(15 * wt / 25) * 25 or 25
        doses = vd.DosingHistory.repeating(dose, 8.0, 12).doses
        eta = rng.multivariate_normal(np.zeros(2), vd.SMIT2021.omega)
        sim = vd.simulate_levels(pt, doses, [26.0, 31.5], eta=eta, residual_error=True,
                                 seed=int(rng.integers(1e6)))
        lv = [vd.Level(t, c) for t, c in zip(sim["times"], sim["observed"])]
        f = vd.fit(pt, doses, lv)
        true_auc = dose * 24.0 / 8.0 / sim["parameters"]["CL"]
        est_auc = dose * 24.0 / 8.0 / f.parameters["CL"]
        errors.append(100.0 * (est_auc - true_auc) / true_auc)
    mpe = float(np.mean(errors))
    mape = float(np.mean(np.abs(errors)))
    assert abs(mpe) < 10.0, f"bias too large: {mpe:.1f}%"
    assert mape < 20.0, f"imprecision too large: {mape:.1f}%"


def test_level_before_first_dose_is_rejected():
    pt = _patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 4, start=0.0).doses
    with pytest.raises(ValueError):
        vd.fit(pt, doses, [vd.Level(-1.0, 10.0)])


# ---------------------------------------------------------------- dosing
def test_dose_follows_target_auc_and_clearance():
    pt = _patient()
    rec = vd.initial_dose(pt, target_auc24=500.0)
    cl = vd.SMIT2021.typical(pt)["CL"]
    assert rec.best.total_daily_dose_mg == pytest.approx(500.0 * cl, rel=0.08)
    assert rec.best.auc24 == pytest.approx(500.0, rel=0.08)


def test_interval_lengthens_as_renal_function_falls():
    good = vd.start_dose(weight_kg=50, age_years=15, height_cm=165, scr=0.6)
    poor = vd.start_dose(weight_kg=50, age_years=15, height_cm=165, scr=2.0)
    assert poor.best.interval_h >= good.best.interval_h
    assert poor.best.total_daily_dose_mg < good.best.total_daily_dose_mg


def test_dose_ceilings_are_respected():
    rec = vd.start_dose(weight_kg=150, age_years=17, height_cm=175, scr=0.5,
                        )
    assert rec.best.dose_mg <= 2000.0
    assert rec.best.total_daily_dose_mg <= 4500.0 + 1e-9


def test_round_dose_and_infusion_duration():
    assert vd.round_dose(312.4) == 300.0
    assert vd.round_dose(47.0) == 45.0
    assert vd.round_dose(1234.0) == 1250.0
    assert vd.infusion_duration(500) == 1.0         # 500 mg/h is under the rate ceiling
    assert vd.infusion_duration(1800) == 3.0        # 10 mg/min ceiling forces 3 h


def test_predicted_exposure_matches_tdd_over_cl():
    pt = _patient()
    f = vd.fit(pt, [], [])
    e = vd.predict_exposure(f, 300.0, 6.0, 1.0)
    assert e.auc24 == pytest.approx(1200.0 / f.parameters["CL"], rel=1e-9)
    assert e.ctrough < e.cmax


def test_assessment_end_to_end():
    a = vd.assess(
        patient={"weight_kg": 20, "age_years": 6, "height_cm": 115, "scr": 0.4},
        regimen={"dose_mg": 300, "interval_h": 6, "n_doses": 4, "start": "2026-01-01 08:00"},
        levels=[("2026-01-01 23:00", 9.4)],
    )
    assert a.auc24_now > 0
    assert a.fit.parameters["CL"] > 0
    assert "Recommended" in a.summary()
    # the recommendation should put the predicted AUC inside the target window
    assert 400 <= a.recommendation.best.auc24 <= 600


# ---------------------------------------------------------------- nomogram
def test_nomogram_matches_published_table():
    pt = vd.Patient(weight_kg=20.0, crcl=100.0)
    n = smit_nomogram(pt)
    assert (n["mg_per_kg_per_dose"], n["interval_h"]) == (15.0, 6.0)
    assert smit_nomogram(vd.Patient(weight_kg=50.0, crcl=70.0))["mg_per_kg_per_dose"] == 11.0
    assert smit_nomogram(vd.Patient(weight_kg=90.0, crcl=120.0))["interval_h"] == 12.0
    assert smit_nomogram(vd.Patient(weight_kg=90.0, crcl=40.0))["mg_per_kg_per_dose"] == 6.0


def test_nomogram_reproduces_published_day3_exposures():
    """Typical individuals dosed per Table III must land in the paper's 400-700 window."""
    aucs, cmins = [], []
    for wt, ht in [(11, 78), (18, 108), (30, 135), (50, 160), (70, 170), (100, 175), (150, 180)]:
        for crcl in (10.1, 40.0, 70.0, 120.0):
            pt = vd.Patient(weight_kg=wt, height_cm=ht, crcl=crcl)
            n = smit_nomogram(pt)
            tau, d = n["interval_h"], n["dose_mg"]
            doses = [Dose(n["first_dose_mg"] if i == 0 else d, i * tau, 1.0)
                     for i in range(int(96 / tau) + 1)]
            pk = model_from_params(vd.SMIT2021.typical(pt))
            aucs.append(pk.auc(doses, 48.0, 72.0))
            cmins.append(float(pk.concentrations(doses, [72.0])[0]))
    assert min(aucs) >= 400.0 and max(aucs) <= 700.0
    assert 7.0 <= min(cmins) and max(cmins) <= 24.0


# ---------------------------------------------------------------- reference method
def test_sawchuk_zaske_recovers_a_one_compartment_truth():
    """Simulate a one-compartment patient, then check the two-level method recovers it."""
    CL, V = 3.0, 25.0
    pk = TwoCompartmentModel(CL=CL, V1=V, Q=1e-9, V2=1e-6)
    dose, tau, T = 600.0, 8.0, 1.0
    doses = [Dose(dose, i * tau, T) for i in range(60)]
    t0 = 50 * tau
    t1, t2 = 2.0, 7.5                     # hours after the start of that infusion
    c1, c2 = pk.concentrations(doses, [t0 + t1, t0 + t2])
    res = vd.sawchuk_zaske(dose, tau, T, t1, c1, t2, c2)
    assert res.clearance_l_h == pytest.approx(CL, rel=0.02)
    assert res.volume_l == pytest.approx(V, rel=0.05)
    assert res.auc24 == pytest.approx(dose * 24 / tau / CL, rel=0.02)


def test_sawchuk_zaske_rejects_level_during_infusion():
    with pytest.raises(ValueError):
        vd.sawchuk_zaske(600, 8, 1.0, 0.5, 30.0, 7.0, 10.0)


def test_mpe_mape():
    out = vd.mpe_mape([100, 200], [110, 180])
    assert out["mpe"] == pytest.approx((10 - 10) / 2)
    assert out["mape"] == pytest.approx(10.0)


# ---------------------------------------------------------------- input handling
def test_model_requires_its_own_covariates():
    """A Patient may be sparsely populated; the chosen model says what it needs."""
    pt = vd.Patient(weight_kg=20.0, age_years=6.0)          # no creatinine, no height
    with pytest.raises(ValueError, match="crcl"):
        vd.fit(pt, doses=[], levels=[], model="smit2021")
    with pytest.raises(ValueError, match="scr_mg_dl"):
        vd.fit(pt, doses=[], levels=[], model="le2013")


def test_out_of_range_covariates_warn_not_fail():
    pt = vd.Patient(weight_kg=4.0, age_years=0.3, height_cm=55, scr_mg_dl=0.3)
    msgs = vd.SMIT2021.check(pt)
    assert any("age_years" in m for m in msgs) and any("weight_kg" in m for m in msgs)
    assert all("extrapolation" in m for m in msgs)
    # warnings, not failures: the fit still runs
    assert vd.fit(pt, doses=[], levels=[], model="smit2021").parameters["CL"] > 0


def test_negative_or_zero_levels_rejected():
    with pytest.raises(ValueError):
        vd.Level(time_h=5.0, conc_mg_l=0.0)


def test_datetime_and_numeric_times_agree():
    pat = {"weight_kg": 20, "age_years": 6, "height_cm": 115, "scr": 0.4}
    a = vd.assess(pat, {"dose_mg": 300, "interval_h": 6, "n_doses": 4,
                        "start": "2026-01-01 08:00"},
                  levels=[("2026-01-01 23:00", 9.4)])
    b = vd.assess(pat, {"dose_mg": 300, "interval_h": 6, "n_doses": 4, "start": 0.0},
                  levels=[(15.0, 9.4)])
    assert a.fit.parameters["CL"] == pytest.approx(b.fit.parameters["CL"], rel=1e-6)


# ---------------------------------------------------------------- structure
def test_one_compartment_solver_matches_closed_form():
    from vankedoodle.pk import OneCompartmentModel
    CL, V = 2.0, 10.0
    pk = OneCompartmentModel(CL=CL, V=V)
    assert pk.n_compartments == 1
    ke = CL / V
    T, dose = 1.0, 500.0
    for t in (0.5, 1.0, 4.0):
        got = pk.concentrations([Dose(dose, 0.0, T)], [t])[0]
        want = (dose / T) / CL * (1 - math.exp(-ke * min(t, T)))
        if t > T:
            want *= math.exp(-ke * (t - T))
        assert got == pytest.approx(want, rel=1e-8)
    assert pk.terminal_half_life == pytest.approx(math.log(2) / ke)
    assert pk.auc([Dose(dose, 0.0, T)], 0.0, 5000.0) == pytest.approx(dose / CL, rel=1e-6)


def test_model_from_params_dispatches_on_structure():
    from vankedoodle.pk import model_from_params
    assert model_from_params({"CL": 2.0, "V": 10.0}).n_compartments == 1
    assert model_from_params({"CL": 2.0, "V1": 8.0, "Q": 1.5, "V2": 12.0}).n_compartments == 2
    with pytest.raises(ValueError):
        model_from_params({"CL": 2.0})


def test_one_compartment_steady_state_auc():
    from vankedoodle.pk import OneCompartmentModel
    pk = OneCompartmentModel(CL=2.5, V=15.0)
    doses = [Dose(400, i * 8.0, 1.0) for i in range(60)]
    assert pk.auc(doses, 50 * 8.0, 51 * 8.0) == pytest.approx(400.0 / 2.5, rel=1e-5)


# ---------------------------------------------------------------- model library
def test_registry_contents():
    assert set(vd.MODELS) >= {"smit2021", "xu2026", "le2013"}
    assert vd.get_model("le2013").n_compartments == 1
    assert vd.get_model("xu2026").n_compartments == 2
    with pytest.raises(KeyError):
        vd.get_model("not_a_model")


def test_le2013_reproduces_published_equation():
    """At Wt = 1 kg, SCr = 0.48 mg/dL and ln(age_days) = 7.8 every term is 1."""
    pt = vd.Patient(weight_kg=1.0, scr_mg_dl=0.48,
                    postnatal_age_days=math.exp(7.8))
    p = vd.LE2013.typical(pt)
    assert p["CL"] == pytest.approx(0.248, rel=1e-9)
    assert p["V"] == pytest.approx(0.636, rel=1e-9)
    # covariate directions: higher creatinine and younger age lower clearance
    older = vd.Patient(weight_kg=1.0, scr_mg_dl=0.48, postnatal_age_days=math.exp(7.8) * 2)
    renal = vd.Patient(weight_kg=1.0, scr_mg_dl=0.96, postnatal_age_days=math.exp(7.8))
    assert vd.LE2013.typical(older)["CL"] > p["CL"]
    assert vd.LE2013.typical(renal)["CL"] < p["CL"]
    assert vd.LE2013.typical(renal)["CL"] == pytest.approx(0.248 * 0.5 ** 0.361, rel=1e-9)


def test_le2013_variances_are_as_tabulated():
    assert vd.LE2013.omega[0, 0] == pytest.approx(0.125)
    assert vd.LE2013.omega[1, 1] == pytest.approx(0.031)
    assert vd.LE2013.sigma_prop == pytest.approx(math.sqrt(0.0857))


def test_xu2026_reproduces_published_equation():
    """At 70 kg, adult PMA and SCr equal to the age-adjusted reference."""
    from vankedoodle.models import xu_scr_std_umol_l
    age = 18.0
    pt = vd.Patient(weight_kg=70.0, age_years=age,
                    scr_umol_l=xu_scr_std_umol_l(age))
    p = vd.XU2026.typical(pt)
    pma = pt.pma_weeks
    f_mat = pma ** 3.4 / (pma ** 3.4 + 47.7 ** 3.4)
    assert p["CL"] == pytest.approx(7.56 * f_mat, rel=1e-9)
    assert f_mat > 0.99                      # clearance is mature by adolescence
    assert p["V1"] == pytest.approx(85.6, rel=1e-9)
    assert p["Q"] == pytest.approx(1.30, rel=1e-9)
    assert p["V2"] == pytest.approx(286.0, rel=1e-9)


def test_xu2026_maturation_is_half_at_tm50():
    from vankedoodle.models import xu_scr_std_umol_l
    # PMA of exactly 47.7 weeks: a 40-week-gestation infant aged 7.7 weeks
    pt = vd.Patient(weight_kg=4.0, gestational_age_weeks=40.0,
                    postnatal_age_days=7.7 * 7.0, scr_umol_l=30.0)
    assert pt.pma_weeks == pytest.approx(47.7)
    f_mat = pt.pma_weeks ** 3.4 / (pt.pma_weeks ** 3.4 + 47.7 ** 3.4)
    assert f_mat == pytest.approx(0.5)
    expected = 7.56 * (4.0 / 70.0) ** 0.75 * 0.5 * math.exp(
        -0.0188 * (30.0 - xu_scr_std_umol_l(pt.age_years)))
    assert vd.XU2026.typical(pt)["CL"] == pytest.approx(expected, rel=1e-9)


def test_xu2026_creatinine_effect_direction_and_units():
    from vankedoodle.models import xu_scr_std_umol_l
    age = 6.0
    ref = xu_scr_std_umol_l(age)
    low = vd.Patient(weight_kg=20.0, age_years=age, scr_umol_l=ref)
    high = vd.Patient(weight_kg=20.0, age_years=age, scr_umol_l=ref + 50.0)
    assert vd.XU2026.typical(high)["CL"] < vd.XU2026.typical(low)["CL"]
    assert vd.XU2026.typical(high)["CL"] / vd.XU2026.typical(low)["CL"] == \
        pytest.approx(math.exp(-0.0188 * 50.0))
    # the reference rises with age, as serum creatinine does in children
    assert xu_scr_std_umol_l(2.0) < xu_scr_std_umol_l(10.0) < xu_scr_std_umol_l(17.0)


def test_creatinine_units_round_trip():
    pt = vd.Patient(weight_kg=20.0, scr_mg_dl=0.5)
    assert pt.scr_umol_l == pytest.approx(0.5 * 88.4)
    pt2 = vd.Patient(weight_kg=20.0, scr_umol_l=44.2)
    assert pt2.scr_mg_dl == pytest.approx(0.5)


def test_postmenstrual_age():
    pt = vd.Patient(weight_kg=1.2, gestational_age_weeks=28.0, postnatal_age_days=14.0)
    assert pt.pma_weeks == pytest.approx(30.0)
    assert pt.assumed_term_gestation is False
    older = vd.Patient(weight_kg=20.0, age_years=6.0)
    assert older.pma_weeks == pytest.approx(40.0 + 6.0 * 52.1775)
    assert older.assumed_term_gestation is True


# ---------------------------------------------------------------- across models
def _multi_model_patient():
    return vd.Patient(weight_kg=20.0, age_years=6.0, height_cm=115.0, scr_mg_dl=0.4)


def test_models_agree_on_clearance_within_a_factor():
    """Independent models of the same population should not disagree wildly."""
    pt = _multi_model_patient()
    cls = [vd.get_model(n).typical(pt)["CL"] for n in ("smit2021", "xu2026", "le2013")]
    assert max(cls) / min(cls) < 1.5


def test_each_model_fits_and_recommends():
    pt = _multi_model_patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 6).doses
    for name in ("smit2021", "xu2026", "le2013"):
        f = vd.fit(pt, doses, [vd.Level(23.5, 9.0)], model=name)
        assert f.parameters["CL"] > 0
        rec = vd.recommend(f, target_auc24=500.0)
        assert 400 <= rec.best.auc24 <= 600
        assert rec.best.dose_mg > 0


def test_one_compartment_model_recovers_a_known_individual():
    pt = _multi_model_patient()
    doses = vd.DosingHistory.repeating(300, 6.0, 8).doses
    times = [5.5, 11.5, 23.5, 29.5]
    sim = vd.simulate_levels(pt, doses, times, model="le2013", eta=[0.30, -0.15],
                             residual_error=False)
    lv = [vd.Level(t, c) for t, c in zip(times, sim["observed"])]
    f = vd.fit(pt, doses, lv, model="le2013")
    # The MAP objective for a proportional error model carries a log-variance term,
    # which biases the mode slightly toward higher clearance (lower predictions).
    # The bias grows with the residual error; Le 2013 has a 29% proportional error,
    # so a few percent is expected even from noise-free data.
    assert f.parameters["CL"] == pytest.approx(sim["parameters"]["CL"], rel=0.10)
    assert f.parameters["V"] == pytest.approx(sim["parameters"]["V"], rel=0.15)
    assert f.parameters["CL"] >= sim["parameters"]["CL"]


def test_compare_models_tabulates_every_model():
    rows = vd.compare_models(
        {"weight_kg": 20, "age_years": 6, "height_cm": 115, "scr": 0.4},
        {"dose_mg": 300, "interval_h": 6, "n_doses": 5, "start": 0.0},
        levels=[(23.5, 9.0)])
    assert {r["model"] for r in rows} == set(vd.MODELS)
    assert all("error" not in r for r in rows)
    assert all(r["auc24_current"] > 0 for r in rows)


def test_compare_models_reports_missing_covariates_rather_than_failing():
    rows = vd.compare_models({"weight_kg": 20, "age_years": 6, "crcl": 100},
                             {"dose_mg": 300, "interval_h": 6, "n_doses": 5})
    by = {r["model"]: r for r in rows}
    assert "error" not in by["smit2021"]          # crcl supplied directly
    assert "scr_mg_dl" in by["le2013"]["error"]   # creatinine was never given


# ---------------------------------------------------------------- interval rule
def test_default_interval_follows_weight_and_renal_function():
    from vankedoodle.dosing import default_interval
    assert default_interval(vd.Patient(weight_kg=20.0, crcl=100.0)) == 6.0
    assert default_interval(vd.Patient(weight_kg=50.0, crcl=100.0)) == 8.0
    assert default_interval(vd.Patient(weight_kg=90.0, crcl=100.0)) == 12.0
    assert default_interval(vd.Patient(weight_kg=20.0, crcl=20.0)) == 12.0
    assert default_interval(vd.Patient(weight_kg=20.0, crcl=12.0)) == 24.0


def test_explicit_interval_is_honoured():
    pt = _multi_model_patient()
    f = vd.fit(pt, [], [])
    rec = vd.recommend(f, target_auc24=500.0, interval_h=12.0)
    assert rec.best.interval_h == 12.0
    assert len(rec.options) == 1


def test_day_exposure_reported_and_consistent():
    pt = _multi_model_patient()
    f = vd.fit(pt, [], [])
    e = vd.predict_exposure(f, 300.0, 6.0, 1.0)
    assert e.day == 3
    assert e.ctrough < e.cmax
    # by day 3 a one-compartment patient is at steady state, so the day AUC
    # must equal the steady-state value
    f1 = vd.fit(pt, [], [], model="le2013")
    e1 = vd.predict_exposure(f1, 300.0, 6.0, 1.0)
    assert e1.auc24_day == pytest.approx(e1.auc24, rel=0.02)


# ---------------------------------------------------------------- Lamarre 2000
def test_schwartz_k_selection():
    from vankedoodle.core import schwartz_1976, schwartz_k
    assert schwartz_k(6.0) == 0.55
    assert schwartz_k(15.0, sex="M") == 0.70
    assert schwartz_k(15.0, sex="F") == 0.55
    assert schwartz_k(15.0, sex=None) == 0.55          # unknown sex takes the lower value
    assert schwartz_k(0.5) == 0.45                      # term infant
    assert schwartz_k(0.5, gestational_age_weeks=30.0) == 0.33   # preterm infant
    assert schwartz_1976(100.0, 0.5, 6.0) == pytest.approx(0.55 * 100 / 0.5)


def test_body_surface_area_formula():
    from vankedoodle.core import body_surface_area
    assert body_surface_area(30.0) == pytest.approx((4 * 30 + 7) / (30 + 90))


def test_lamarre_clearance_is_slope_times_clcr_plus_intercept():
    from vankedoodle.models import lamarre_creatinine_clearance
    pt = vd.Patient(weight_kg=30.0, age_years=8.0, height_cm=130.0, scr_mg_dl=0.5)
    ccl = lamarre_creatinine_clearance(pt)
    p = vd.LAMARRE2000.typical(pt)
    assert p["CL"] == pytest.approx(0.46 * ccl + 0.018 * 30.0, rel=1e-12)
    assert p["V1"] == pytest.approx(0.27 * 30.0)
    assert p["V2"] == pytest.approx(0.16 * 30.0)
    assert p["Q"] == pytest.approx(0.16 * 30.0)


def test_lamarre_creatinine_clearance_units():
    """CCL is the Schwartz clearance in L/h, de-normalised to the patient's BSA."""
    from vankedoodle.core import body_surface_area, schwartz_1976
    from vankedoodle.models import lamarre_creatinine_clearance
    pt = vd.Patient(weight_kg=30.0, age_years=8.0, height_cm=130.0, scr_mg_dl=0.5)
    expect = (schwartz_1976(130.0, 0.5, 8.0) * 0.06
              * body_surface_area(30.0) / 1.73)
    assert lamarre_creatinine_clearance(pt) == pytest.approx(expect, rel=1e-12)
    # a 30 kg child with normal renal function lands near the published mean CL
    assert vd.LAMARRE2000.typical(pt)["CL"] / 30.0 == pytest.approx(0.10, abs=0.03)


def test_lamarre_has_three_random_effects():
    pt = vd.Patient(weight_kg=30.0, age_years=8.0, height_cm=130.0, scr_mg_dl=0.5)
    assert vd.LAMARRE2000.n_eta == 3
    doses = vd.DosingHistory.repeating(450, 8.0, 6).doses
    f = vd.fit(pt, doses, [vd.Level(23.5, 8.0)], model="lamarre2000")
    assert f.eta.shape == (3,)
    assert f.eta_covariance.shape == (3, 3)


# ---------------------------------------------------------------- Colin 2019
def test_colin_maturation_and_decline_are_half_at_their_midpoints():
    from vankedoodle.models import (COLIN_AGE50_YEARS, COLIN_GAMMA1, COLIN_GAMMA2,
                                    COLIN_PMA50_WEEKS)
    pma_wk = COLIN_PMA50_WEEKS
    f_mat = pma_wk ** COLIN_GAMMA1 / (pma_wk ** COLIN_GAMMA1 + COLIN_PMA50_WEEKS ** COLIN_GAMMA1)
    assert f_mat == pytest.approx(0.5)
    age = COLIN_AGE50_YEARS
    f_dec = COLIN_AGE50_YEARS ** COLIN_GAMMA2 / (COLIN_AGE50_YEARS ** COLIN_GAMMA2
                                                 + age ** COLIN_GAMMA2)
    assert f_dec == pytest.approx(0.5)


def test_colin_reaches_90_percent_of_adult_clearance_by_2_years_pma():
    """The paper states 90% of adult clearance is reached by 2 years PMA."""
    from vankedoodle.models import COLIN_GAMMA1, COLIN_PMA50_WEEKS
    pma_wk = 2.0 * 52.1775
    f_mat = pma_wk ** COLIN_GAMMA1 / (pma_wk ** COLIN_GAMMA1 + COLIN_PMA50_WEEKS ** COLIN_GAMMA1)
    assert f_mat == pytest.approx(0.90, abs=0.02)


def test_colin_creatinine_sensitivity_matches_the_paper():
    """Published: SCr 0.20 mg/dL below/above SCRstd changes CL by +13.8%/-12.2%."""
    from vankedoodle.models import colin_scr_std_mg_dl
    pma_yr = 40.0
    std = colin_scr_std_mg_dl(pma_yr)
    pma_wk = pma_yr * 52.1775
    base = vd.Patient(weight_kg=70.0, scr_mg_dl=std, age_years=pma_yr - 40.0 / 52.1775,
                      postnatal_age_days=(pma_wk - 40.0) * 7.0)
    low = vd.Patient(weight_kg=70.0, scr_mg_dl=std - 0.20,
                     postnatal_age_days=(pma_wk - 40.0) * 7.0)
    high = vd.Patient(weight_kg=70.0, scr_mg_dl=std + 0.20,
                      postnatal_age_days=(pma_wk - 40.0) * 7.0)
    cl0 = vd.COLIN2019.typical(base)["CL"]
    assert vd.COLIN2019.typical(low)["CL"] / cl0 == pytest.approx(1.138, abs=0.002)
    assert vd.COLIN2019.typical(high)["CL"] / cl0 == pytest.approx(0.878, abs=0.002)


def test_colin_typical_volumes_per_70_kg():
    from vankedoodle.models import colin_scr_std_mg_dl
    pma_wk = 40.0 * 52.1775
    pt = vd.Patient(weight_kg=70.0, postnatal_age_days=(pma_wk - 40.0) * 7.0,
                    scr_mg_dl=colin_scr_std_mg_dl(pma_wk / 52.1775))
    p = vd.COLIN2019.typical(pt)
    assert p["V1"] == pytest.approx(42.9)
    assert p["V2"] == pytest.approx(41.7)
    assert p["Q"] == pytest.approx(3.22)


def test_colin_uses_combined_residual_error():
    assert vd.COLIN2019.sigma_prop == pytest.approx(0.215)
    assert vd.COLIN2019.sigma_add == pytest.approx(1.23)


# ---------------------------------------------------------------- Frymoyer 2014
def test_frymoyer_reproduces_published_equation():
    from vankedoodle.models import FRYMOYER_HILL, FRYMOYER_TM50_WEEKS
    # at the reference weight, SCr = 1 mg/dL and PMA = TM50 every term is known
    pt = vd.Patient(weight_kg=2.9, gestational_age_weeks=34.8, postnatal_age_days=0.0,
                    scr_mg_dl=1.0)
    assert pt.pma_weeks == pytest.approx(FRYMOYER_TM50_WEEKS)
    p = vd.FRYMOYER2014.typical(pt)
    assert p["CL"] == pytest.approx(0.345 * 0.5, rel=1e-9)
    assert p["V"] == pytest.approx(1.75, rel=1e-9)


def test_frymoyer_typical_clearance_at_the_cohort_median():
    """Median neonate: 2.9 kg, PMA 39 weeks, SCr 0.4 mg/dL."""
    pt = vd.Patient(weight_kg=2.9, gestational_age_weeks=37.0,
                    postnatal_age_days=2.0 * 7.0, scr_mg_dl=0.4)
    assert pt.pma_weeks == pytest.approx(39.0)
    assert vd.FRYMOYER2014.typical(pt)["CL"] == pytest.approx(0.276, abs=0.002)


def test_frymoyer_clearance_rises_with_maturation_and_falls_with_creatinine():
    young = vd.Patient(weight_kg=2.0, gestational_age_weeks=28.0,
                       postnatal_age_days=7.0, scr_mg_dl=0.8)
    older = vd.Patient(weight_kg=2.0, gestational_age_weeks=38.0,
                       postnatal_age_days=7.0, scr_mg_dl=0.8)
    renal = vd.Patient(weight_kg=2.0, gestational_age_weeks=38.0,
                       postnatal_age_days=7.0, scr_mg_dl=1.6)
    assert (vd.FRYMOYER2014.typical(young)["CL"]
            < vd.FRYMOYER2014.typical(older)["CL"])
    assert (vd.FRYMOYER2014.typical(renal)["CL"]
            < vd.FRYMOYER2014.typical(older)["CL"])


# ---------------------------------------------------------------- neonatal use
def _neonate():
    return vd.Patient(weight_kg=2.9, gestational_age_weeks=36.0,
                      postnatal_age_days=21.0, height_cm=48.0, scr_mg_dl=0.6)


def test_neonatal_models_fit_and_recommend():
    pt = _neonate()
    doses = vd.DosingHistory.repeating(45, 12.0, 6).doses
    for name in ("frymoyer2014", "colin2019"):
        f = vd.fit(pt, doses, [vd.Level(23.0, 11.0)], model=name)
        assert f.parameters["CL"] > 0
        rec = vd.recommend(f, target_auc24=500.0, intervals=(8.0, 12.0, 18.0, 24.0))
        assert rec.best.dose_mg > 0
        assert 400 <= rec.best.auc24 <= 600


def test_registry_now_spans_six_models_and_both_structures():
    assert len(vd.MODELS) == 6
    assert {vd.get_model(n).n_compartments for n in vd.MODELS} == {1, 2}
    rows = vd.compare_models(_neonate(), {"dose_mg": 45, "interval_h": 12, "n_doses": 5})
    assert {r["model"] for r in rows} == set(vd.MODELS)
    ok = [r for r in rows if "error" not in r]
    assert {"frymoyer2014", "colin2019"} <= {r["model"] for r in ok}


# ---------------------------------------------------------------- guards
def test_neonatal_model_warns_when_used_on_an_older_child():
    """A neonatal prior must not answer silently for a 7-year-old."""
    child = vd.Patient(weight_kg=23.0, age_years=7.0, height_cm=122.0, scr_mg_dl=0.33)
    msgs = vd.FRYMOYER2014.check(child)
    assert any("pma_weeks" in m and "extrapolation" in m for m in msgs)
    assert vd.FRYMOYER2014.check(_neonate()) == []      # silent where it belongs


def test_prematurity_warning_only_fires_while_maturation_matters():
    older = vd.Patient(weight_kg=23.0, age_years=7.0, scr_mg_dl=0.33)
    infant = vd.Patient(weight_kg=2.9, postnatal_age_days=21.0, scr_mg_dl=0.6)
    assert not any("Gestational age" in m for m in vd.COLIN2019.check(older))
    assert any("Gestational age" in m for m in vd.COLIN2019.check(infant))
    # supplying gestational age silences it
    known = vd.Patient(weight_kg=2.9, gestational_age_weeks=36.0,
                       postnatal_age_days=21.0, scr_mg_dl=0.6)
    assert not any("Gestational age" in m for m in vd.COLIN2019.check(known))


def test_smit_renal_cap_binds_for_augmented_clearance():
    """Smit caps CLcr at 120, so it cannot represent augmented renal clearance."""
    arc = vd.Patient(weight_kg=23.0, age_years=7.0, height_cm=122.0, scr_mg_dl=0.33)
    assert arc.crcl > 120.0
    capped = vd.Patient(weight_kg=23.0, age_years=7.0, crcl=120.0)
    assert vd.SMIT2021.typical(arc)["CL"] == pytest.approx(vd.SMIT2021.typical(capped)["CL"])
    assert any("crcl" in m and "extrapolation" in m for m in vd.SMIT2021.check(arc))


def test_interval_does_not_change_the_daily_dose():
    """AUC24 depends on the daily dose and clearance, not on how the day is split."""
    pt = vd.Patient(weight_kg=23.0, age_years=7.0, height_cm=122.0, scr_mg_dl=0.33)
    f = vd.fit(pt, [], [], model="colin2019")
    six = vd.recommend(f, target_auc24=500.0, interval_h=6.0).best
    eight = vd.recommend(f, target_auc24=500.0, interval_h=8.0).best
    assert six.total_daily_dose_mg == pytest.approx(eight.total_daily_dose_mg, rel=0.05)
    assert eight.cmax > six.cmax and eight.ctrough < six.ctrough


def test_xu2026_reproduces_published_simulated_troughs():
    """Table 3 of Xu 2026 tabulates median troughs on 15 mg/kg q6h by age and SCr.

    A typical individual from each stratum must land inside the published IQR.
    Representative weights are 50th-percentile values for each age band, so this
    is a check on the covariate model and the solver, not on the exact cohort.
    """
    from vankedoodle.pk import model_from_params
    strata = [   # age (y), weight (kg), SCr (umol/L), published median trough, IQR
        (0.6, 7.5, 20.0, 7.92, (5.34, 10.84)),
        (2.5, 13.0, 20.0, 7.82, (5.51, 10.43)),
        (2.5, 13.0, 40.0, 12.07, (9.07, 15.69)),
        (7.0, 23.0, 30.0, 9.58, (6.84, 13.17)),
        (7.0, 23.0, 50.0, 14.01, (10.53, 18.21)),
        (13.0, 45.0, 45.0, 11.96, (8.36, 15.82)),
        (13.0, 45.0, 70.0, 18.49, (14.06, 23.72)),
        (17.0, 62.0, 65.0, 14.41, (10.71, 19.59)),
        (17.0, 62.0, 95.0, 23.27, (18.16, 29.51)),
    ]
    for age, wt, scr, published, (lo, hi) in strata:
        pt = vd.Patient(weight_kg=wt, age_years=age, scr_umol_l=scr)
        pk = model_from_params(vd.XU2026.typical(pt))
        dose = vd.round_dose(15.0 * wt, increment=1.0)
        doses = [Dose(dose, i * 6.0, 1.0) for i in range(10)]
        trough = float(pk.concentrations(doses, [48.0])[0])   # end of the AUC24-48 window
        assert lo <= trough <= hi, (age, trough, published)


def test_central_volume_moves_peaks_but_not_auc():
    """AUC24 = total daily dose / CL, so V1 changes concentrations and nothing else.

    This is why Xu 2026's large central volume gives low predicted peaks without
    affecting the dose it recommends.
    """
    from vankedoodle.pk import CompartmentalModel
    pt = vd.Patient(weight_kg=23.0, age_years=7.0, height_cm=122.0, scr_mg_dl=0.33)
    pars = vd.XU2026.typical(pt)
    assert pars["V1"] / 23.0 == pytest.approx(85.6 / 70.0, rel=1e-9)   # 1.22 L/kg

    def profile(v1):
        pk = CompartmentalModel.two_compartment(pars["CL"], v1, pars["Q"], pars["V2"])
        doses = [Dose(450.0, i * 8.0, 1.0) for i in range(10)]
        conc = pk.concentrations(doses, np.linspace(64, 72, 401))
        return pk.auc24_steady_state(450.0 * 3), float(conc.max()), float(conc[-1])

    auc_full, peak_full, trough_full = profile(pars["V1"])
    auc_third, peak_third, trough_third = profile(pars["V1"] / 3.0)
    assert auc_third == pytest.approx(auc_full, rel=1e-12)     # AUC is untouched
    assert peak_third > 1.7 * peak_full                        # the peak nearly doubles
    assert trough_third < trough_full


def test_simulated_error_matches_the_models_error_model():
    """simulate_levels must use the same combined error model the fit assumes."""
    pt = vd.Patient(weight_kg=20.0, age_years=6.0, height_cm=115.0, scr_mg_dl=0.4)
    doses = vd.DosingHistory.repeating(300, 6.0, 8).doses
    times = [23.5] * 4000
    # smit2021: proportional only -> relative spread equals sigma_prop
    sim = vd.simulate_levels(pt, doses, times, model="smit2021", eta=[0.0, 0.0], seed=3)
    rel = (sim["observed"] - sim["true"]) / sim["true"]
    assert rel.std() == pytest.approx(vd.SMIT2021.sigma_prop, rel=0.08)
    # colin2019: proportional plus additive -> total variance is the sum
    sim = vd.simulate_levels(pt, doses, times, model="colin2019", eta=[0.0, 0.0, 0.0], seed=3)
    resid = sim["observed"] - sim["true"]
    expect = np.sqrt((vd.COLIN2019.sigma_prop * sim["true"][0]) ** 2
                     + vd.COLIN2019.sigma_add ** 2)
    assert resid.std() == pytest.approx(expect, rel=0.08)
    assert vd.COLIN2019.sigma_add > 0


# ---------------------------------------------------------------- JSON API
import json as _json


def _request(**overrides):
    req = {
        "case_id": "test-001",
        "patient": {"weight_kg": 23.0, "age_years": 7.0, "height_cm": 122.0,
                    "scr_mg_dl": 0.33, "sex": "M"},
        "regimen": {"dose_mg": 450, "interval_h": 8, "n_doses": 5, "infusion_h": 1.0,
                    "start": "2026-03-02 08:00"},
        "levels": [{"time": "2026-03-03 07:30", "conc_mg_l": 6.2},
                   {"time": "2026-03-03 10:00", "conc_mg_l": 30.2},
                   {"time": "2026-03-03 15:30", "conc_mg_l": 8.3}],
        "options": {"model": "smit2021", "series": False},
    }
    req.update(overrides)
    return req


def test_api_response_is_json_serialisable():
    """No numpy types may leak into the response, or json.dumps fails in the UI."""
    resp = vd.evaluate(_request(options={"model": "all", "series": True}))
    assert resp["ok"] is True
    text = _json.dumps(resp)
    assert _json.loads(text)["case_id"] == "test-001"
    assert resp["schema_version"] == vd.SCHEMA_VERSION


def test_api_runs_every_model_and_reports_each():
    resp = vd.evaluate(_request(options={"model": "all", "series": False}))
    assert {r["model"] for r in resp["results"]} == set(vd.MODELS)
    for entry in resp["results"]:
        assert entry["ok"] is True
        assert entry["current"]["auc24"] > 0
        assert entry["recommendation"]["dose_mg"] > 0
        assert isinstance(entry["out_of_population"], bool)


def test_api_returns_user_errors_as_data_not_exceptions():
    """Request-level problems come back as messages, never as tracebacks."""
    no_doses = _request()
    no_doses.pop("regimen")
    resp = vd.evaluate(no_doses)
    assert resp["ok"] is False
    assert any("doses" in e.lower() for e in resp["errors"])

    resp = vd.evaluate(_request(options={"model": "not_a_model"}))
    assert resp["ok"] is False
    assert any("Unknown model" in e for e in resp["errors"])

    resp = vd.evaluate(_request(patient={"weight_kg": 23.0, "nonsense_field": 1}))
    assert resp["ok"] is False
    assert any("nonsense_field" in e for e in resp["errors"])


def test_api_missing_covariate_is_reported_per_model_not_globally():
    """A model that cannot run must not sink the whole response."""
    resp = vd.evaluate(_request(
        patient={"weight_kg": 23.0, "age_years": 7.0, "crcl": 110.0},
        options={"model": "all", "series": False}))
    assert resp["ok"] is True
    by = {r["model"]: r for r in resp["results"]}
    assert by["smit2021"]["ok"] is True            # crcl given directly
    assert by["le2013"]["ok"] is False             # needs serum creatinine
    assert "scr_mg_dl" in by["le2013"]["error"]


def test_api_accepts_explicit_doses_and_hours():
    clock = vd.evaluate(_request(options={"model": "smit2021", "series": False}))
    hours = vd.evaluate(_request(
        regimen={"dose_mg": 450, "interval_h": 8, "n_doses": 5, "infusion_h": 1.0,
                 "start": 0.0},
        levels=[{"time": 23.5, "conc_mg_l": 6.2}, {"time": 26.0, "conc_mg_l": 30.2},
                {"time": 31.5, "conc_mg_l": 8.3}],
        options={"model": "smit2021", "series": False}))
    assert clock["results"][0]["parameters"]["CL"] == pytest.approx(
        hours["results"][0]["parameters"]["CL"], rel=1e-6)

    explicit = vd.evaluate(_request(
        doses=[{"amount_mg": 450, "time": i * 8.0, "infusion_h": 1.0} for i in range(5)],
        levels=[{"time": 23.5, "conc_mg_l": 6.2}, {"time": 26.0, "conc_mg_l": 30.2},
                {"time": 31.5, "conc_mg_l": 8.3}],
        options={"model": "smit2021", "series": False}))
    explicit.pop("regimen", None)
    assert explicit["results"][0]["parameters"]["CL"] == pytest.approx(
        hours["results"][0]["parameters"]["CL"], rel=1e-6)


def test_api_series_are_plot_ready():
    resp = vd.evaluate(_request(options={"model": "smit2021", "series": True}))
    series = resp["results"][0]["series"]
    for key in ("time_h", "conc_prior", "conc_posterior",
                "projection_time_h", "projection_conc"):
        assert len(series[key]) > 50
        assert all(isinstance(v, float) for v in series[key][:5])
    assert len(series["time_h"]) == len(series["conc_posterior"])
    # The projection begins at the last recorded dose, so it overlaps the final
    # infusion and a front end can draw the two curves as one continuous line.
    assert series["projection_time_h"][0] <= max(series["time_h"])
    # ...and runs one recommended interval plus a day past the last dose (32 h)
    rec = resp["results"][0]["recommendation"]
    assert series["projection_time_h"][-1] == pytest.approx(32.0 + rec["interval_h"] + 24.0)


def test_api_candidate_regimen_is_evaluated():
    resp = vd.evaluate(_request(options={
        "model": "smit2021", "series": False,
        "candidate_regimen": {"dose_mg": 600, "interval_h": 8, "infusion_h": 1.0}}))
    cand = resp["results"][0]["candidate"]
    assert cand["dose_mg"] == 600 and cand["interval_h"] == 8
    assert cand["auc24"] > resp["results"][0]["current"]["auc24"]


# ------------------------------------------------- two-level auto-detection
def test_two_level_detection_finds_a_usable_pair():
    resp = vd.evaluate(_request())
    sz = resp["sawchuk_zaske"]
    assert sz["available"] is True
    assert sz["auc24"] > 0 and sz["clearance_l_h"] > 0
    assert len(sz["levels_used"]) == 2
    assert sz["levels_used"][0]["hours_into_interval"] >= 1.0   # after the infusion


def test_two_level_detection_explains_why_it_cannot_run():
    one = vd.evaluate(_request(levels=[{"time": "2026-03-03 07:30", "conc_mg_l": 6.2}]))
    assert one["sawchuk_zaske"]["available"] is False
    assert "Two levels are needed" in one["sawchuk_zaske"]["reason"]
    assert one["ok"] is True                       # the Bayesian result still stands
    assert one["results"][0]["current"]["auc24"] > 0

    # two levels, but in different dosing intervals
    split = vd.evaluate(_request(levels=[{"time": 7.5, "conc_mg_l": 9.0},
                                         {"time": 15.5, "conc_mg_l": 8.6}],
                                 regimen={"dose_mg": 450, "interval_h": 8, "n_doses": 5,
                                          "infusion_h": 1.0, "start": 0.0}))
    assert split["sawchuk_zaske"]["available"] is False
    assert "same dosing interval" in split["sawchuk_zaske"]["reason"]

    # two levels in one interval, but the first is drawn during the infusion
    during = vd.evaluate(_request(levels=[{"time": 24.5, "conc_mg_l": 28.0},
                                          {"time": 29.0, "conc_mg_l": 9.0}],
                                  regimen={"dose_mg": 450, "interval_h": 8, "n_doses": 5,
                                           "infusion_h": 1.0, "start": 0.0}))
    assert during["sawchuk_zaske"]["available"] is False
    assert "end of the" in during["sawchuk_zaske"]["reason"]


def test_two_level_result_matches_the_direct_call():
    resp = vd.evaluate(_request())
    sz = resp["sawchuk_zaske"]
    direct = vd.sawchuk_zaske(dose_mg=450.0, interval_h=8.0, infusion_h=1.0,
                              t1_h=2.0, c1_mg_l=30.2, t2_h=7.5, c2_mg_l=8.3)
    assert sz["auc24"] == pytest.approx(direct.auc24, rel=1e-9)


# ---------------------------------------------------------------- case records
def test_case_record_round_trips_through_a_file(tmp_path):
    resp = vd.evaluate(_request(options={"model": "all", "series": False}))
    row = vd.case_record(resp, model="colin2019", extra={"study_id": "PCH-07"})
    assert row["model"] == "colin2019"
    assert row["study_id"] == "PCH-07"
    assert row["bayesian_auc24"] > 0
    assert row["sawchuk_zaske_available"] is True
    assert row["percent_difference_vs_sawchuk_zaske"] == pytest.approx(
        100.0 * (row["bayesian_auc24"] - row["sawchuk_zaske_auc24"])
        / row["sawchuk_zaske_auc24"], rel=1e-9)

    path = tmp_path / "cases.jsonl"
    vd.append_case(path, row)
    vd.append_case(path, vd.case_record(resp, model="smit2021"))
    back = vd.load_cases(path)
    assert len(back) == 2
    assert [r["model"] for r in back] == ["colin2019", "smit2021"]
    assert back[0]["study_id"] == "PCH-07"


def test_case_record_refuses_a_failed_evaluation():
    resp = vd.evaluate(_request(options={"model": "not_a_model"}))
    with pytest.raises(ValueError):
        vd.case_record(resp)


# ---------------------------------------------------------------- uncertainty output
def _uncertain(**opts):
    return vd.evaluate(_request(options={"model": "smit2021", "series": True,
                                         "uncertainty": True, **opts}))


def test_uncertainty_is_off_by_default_and_costs_nothing():
    entry = vd.evaluate(_request())["results"][0]
    assert "distribution" not in entry["recommendation"]
    assert "forecast" not in entry


def test_bands_contain_their_curves():
    s = _uncertain()["results"][0]["series"]
    for curve, band in (("conc_prior", "band_prior"), ("conc_posterior", "band_posterior")):
        lo, hi, mid = (np.array(s[band]["low"]), np.array(s[band]["high"]),
                       np.array(s[curve]))
        assert len(lo) == len(mid)
        assert np.all(lo <= hi + 1e-9)
        # the MAP curve need not be the pointwise median, but it lies inside the band
        assert np.mean((mid >= lo - 1e-6) & (mid <= hi + 1e-6)) > 0.95


def test_levels_narrow_the_band():
    s = _uncertain()["results"][0]["series"]
    width = lambda b: np.mean(np.array(b["high"]) - np.array(b["low"]))
    assert width(s["band_posterior"]) < width(s["band_prior"])


def test_distribution_agrees_with_the_exposure_it_describes():
    rec = _uncertain()["results"][0]["recommendation"]
    d = rec["distribution"]
    assert d["prob_below_target"] + d["prob_in_target"] + d["prob_above_target"] == \
        pytest.approx(1.0)
    assert d["prob_in_target"] == pytest.approx(rec["prob_in_target"], abs=0.03)
    assert sum(d["histogram"]["density"]) == pytest.approx(1.0)
    assert 400.0 in d["histogram"]["bin_edges"] and 600.0 in d["histogram"]["bin_edges"]
    assert d["auc24_median"] == pytest.approx(rec["auc24"], rel=0.05)


def test_peak_risk_follows_the_threshold():
    low = _uncertain(peak_threshold_mg_l=20.0)["results"][0]["recommendation"]
    high = _uncertain(peak_threshold_mg_l=200.0)["results"][0]["recommendation"]
    assert low["distribution"]["prob_peak_above"] > 0.5
    assert high["distribution"]["prob_peak_above"] == 0.0


def test_every_alternative_carries_a_distribution():
    entry = _uncertain()["results"][0]
    assert entry["alternatives"]
    assert all(a["distribution"]["histogram"] for a in entry["alternatives"])


def test_forecast_follows_the_candidate_when_one_is_given():
    plain = _uncertain()["results"][0]["forecast"]
    assert plain["regimen"] == "recommendation"
    cand = _uncertain(candidate_regimen={"dose_mg": 600, "interval_h": 12})["results"][0]
    f = cand["forecast"]
    assert f["regimen"] == "candidate" and f["dose_mg"] == 600 and f["interval_h"] == 12
    assert f["time_h"][-1] == pytest.approx(f["forecast_start_h"] + 72.0)
    # the history part of the forecast reproduces the fitted curve
    s = cand["series"]
    t = np.array(s["time_h"])
    hist = t < f["forecast_start_h"] + 12.0 - 1e-6
    # (compared by median, since linear interpolation clips the sharp peaks)
    a = np.interp(t[hist], f["time_h"], f["conc"])
    b = np.array(s["conc_posterior"])[hist]
    assert np.median(np.abs(a - b)) < 0.2
    assert np.all(np.array(f["low"]) <= np.array(f["high"]) + 1e-9)


def test_posterior_cv_is_smaller_than_prior_cv():
    for row in _uncertain()["results"][0]["parameter_uncertainty"]:
        assert row["posterior_cv_pct"] < row["prior_cv_pct"]


def test_prior_only_case_has_no_posterior_band_or_cv():
    entry = vd.evaluate(_request(levels=[], options={"model": "smit2021", "series": True,
                                                     "uncertainty": True}))["results"][0]
    assert entry["series"]["band_posterior"] is None
    assert entry["series"]["band_prior"] is not None
    assert all(r["posterior_cv_pct"] is None for r in entry["parameter_uncertainty"])


def test_one_compartment_curve_passes_through_the_levels_it_used():
    sz = vd.evaluate(_request())["sawchuk_zaske"]
    c = sz["curve"]
    for lv in sz["levels_used"]:
        assert np.interp(lv["hours_into_interval"], c["time_into_interval_h"],
                         c["conc"]) == pytest.approx(lv["conc_mg_l"], rel=1e-3)


def test_steady_state_interval_matches_the_reported_peak_and_trough():
    entry = _uncertain()["results"][0]
    ss, cur = entry["steady_state_interval"], entry["current"]
    assert max(ss["conc"]) == pytest.approx(cur["cmax"], rel=1e-2)
    assert ss["conc"][-1] == pytest.approx(cur["ctrough"], rel=1e-6)


def test_uncertainty_output_is_json_serialisable_for_every_model():
    resp = _uncertain(model="all")
    _json.dumps(resp)
    assert all(r["ok"] for r in resp["results"])


# ---------------------------------------------------------------- birthdate
def test_age_from_birthdate():
    a = vd.age_from_birthdate("2019-03-02", on="2026-03-02")
    assert a["age_days"] == 2557
    assert a["age_years"] == pytest.approx(7.0, abs=0.01)
    with pytest.raises(ValueError, match="after"):
        vd.age_from_birthdate("2026-03-03", on="2026-03-02")


def test_patient_birthdate_sets_both_ages():
    p = vd.Patient(weight_kg=4.0, birthdate="2026-02-01", age_on="2026-03-02",
                   gestational_age_weeks=36.0)
    assert p.postnatal_age_days == 29
    assert p.age_years == pytest.approx(29 / 365.25)
    assert p.pma_weeks == pytest.approx(36.0 + 29 / 7)


def test_birthdate_gives_the_same_fit_as_the_age_it_implies():
    by_age = vd.evaluate(_request(options={"model": "le2013", "series": False}))
    req = _request(options={"model": "le2013", "series": False})
    days = vd.age_from_birthdate("2019-03-02", on="2026-03-02")["age_days"]
    req["patient"] = {**req["patient"], "age_years": None}
    del req["patient"]["age_years"]
    req["patient"]["birthdate"] = "2019-03-02"
    by_dob = vd.evaluate(req)
    assert by_dob["patient"]["age_on"] == "2026-03-02"       # the first dose date
    assert by_dob["patient"]["postnatal_age_days"] == days
    assert by_dob["results"][0]["parameters"]["CL"] == pytest.approx(
        vd.evaluate(_request(patient={**_request()["patient"], "age_years": None,
                                      "postnatal_age_days": days},
                             options={"model": "le2013", "series": False})
                    )["results"][0]["parameters"]["CL"], rel=1e-9)
    assert by_age["ok"] and by_dob["ok"]


def test_birthdate_after_the_first_dose_is_reported_not_raised():
    req = _request()
    req["patient"] = {k: v for k, v in req["patient"].items() if k != "age_years"}
    req["patient"]["birthdate"] = "2026-04-01"
    resp = vd.evaluate(req)
    assert resp["ok"] is False
    assert "after" in resp["errors"][0]


# ---------------------------------------------------------------- plotting grids
def test_time_grid_contains_every_infusion_start_and_end():
    from vankedoodle.pk import time_grid
    doses = [vd.Dose(amount_mg=100, start_h=8.0 * i, duration_h=1.0) for i in range(3)]
    g = time_grid(0.0, 20.0, doses, n_points=7)
    for t in (0.0, 1.0, 8.0, 9.0, 16.0, 17.0):
        assert t in g
    assert np.all(np.diff(g) > 0)


def test_series_reaches_the_true_peak():
    entry = vd.evaluate(_request(options={"model": "smit2021", "series": True}))["results"][0]
    s = entry["series"]
    t = np.array(s["time_h"])
    for end in (1.0, 9.0, 17.0, 25.0, 33.0):            # each infusion end
        assert end in t


def test_forecast_starts_when_the_next_dose_is_due():
    entry = _uncertain(candidate_regimen={"dose_mg": 600, "interval_h": 12})["results"][0]
    f = entry["forecast"]
    # given every 8 h, last dose at 32 h: the switch happens at 40 h, not 32 + 12
    assert f["next_dose_h"] == pytest.approx(40.0)
    assert f["dose_times_h"][5] == pytest.approx(40.0)
    assert f["dose_times_h"][6] == pytest.approx(52.0)


# ---------------------------------------------------------------- continuous infusion
def test_continuous_infusion_targets_the_auc_and_runs_all_day():
    entry = _uncertain(continuous_infusion=True)["results"][0]
    ci = entry["continuous_infusion"]
    assert ci["continuous"] is True
    assert ci["interval_h"] == 24.0 and ci["infusion_h"] == 24.0
    assert ci["dose_mg"] == pytest.approx(500.0 * entry["parameters"]["CL"], abs=25.0)
    assert ci["auc24"] == pytest.approx(500.0, rel=0.03)
    # no peaks: by day 3 the end-of-day concentration is near AUC24 / 24
    assert ci["cmax"] == pytest.approx(ci["auc24"] / 24.0, rel=0.25)
    assert not any("trough" in f for f in ci["flags"])
    assert ci["distribution"]["prob_peak_above"] == 0.0


def test_continuous_infusion_is_off_by_default():
    assert _uncertain()["results"][0]["continuous_infusion"] is None


def test_a_candidate_infusion_filling_its_interval_is_continuous():
    c = _uncertain(candidate_regimen={"dose_mg": 1300, "interval_h": 24,
                                      "infusion_h": 24})["results"][0]["candidate"]
    assert c["continuous"] is True and c["infusion_h"] == 24.0
    short = _uncertain(candidate_regimen={"dose_mg": 450, "interval_h": 8,
                                          "infusion_h": 1})["results"][0]["candidate"]
    assert short["continuous"] is False


# ---------------------------------------------------------------- posterior sampling
def _single_trough_fit(model="smit2021"):
    from vankedoodle import bayes
    req = _request()
    hist = vd.api._build_history(req)
    pat = vd.api._build_patient(req["patient"], hist.origin)
    lv = req["levels"][0]
    level = vd.Level(time_h=vd.to_hours(lv["time"], hist.origin) if isinstance(lv["time"], str)
                     else float(lv["time"]), conc_mg_l=lv["conc_mg_l"])
    return bayes.fit(pat, hist.doses, [level], model=model)


def _grid_auc_quantiles(fit, tdd, qs=(0.05, 0.5, 0.95), n=160, width=6.0):
    """The exact posterior of AUC24 by brute-force integration over a 2-D eta grid:
    no sampling and no normal approximation, so an independent reference."""
    sd = np.sqrt(np.diag(fit.eta_covariance))
    axes = [np.linspace(m - width * s, m + width * s, n) for m, s in zip(fit.eta, sd)]
    g1, g2 = np.meshgrid(*axes, indexing="ij")
    etas = np.column_stack([g1.ravel(), g2.ravel()])
    logp = -0.5 * np.array([fit._objective(e) for e in etas])
    w = np.exp(logp - logp.max())
    w /= w.sum()
    auc = tdd / np.array([fit.model.individual(fit.patient, e)["CL"] for e in etas])
    order = np.argsort(auc)
    cum = np.cumsum(w[order])
    return [float(auc[order][np.searchsorted(cum, q)]) for q in qs]


def test_auc_interval_matches_the_exact_posterior():
    fit = _single_trough_fit()
    assert fit.posterior_method == "importance sampling"
    tdd = 1350.0
    lo, med, hi = _grid_auc_quantiles(fit, tdd)
    exp = vd.predict_exposure(fit, 450.0, 8.0, 1.0)
    assert exp.auc24_ci[0] == pytest.approx(lo, rel=0.03)
    assert exp.auc24_ci[1] == pytest.approx(hi, rel=0.03)
    assert exp.auc24 == pytest.approx(med, rel=0.03)        # MAP ~ posterior median


def test_the_normal_approximation_alone_would_bias_the_interval_low():
    """Why sampling is needed: with one trough, symmetric normal draws put the
    lower bound well below the exact one."""
    fit = _single_trough_fit()
    lo, _, _ = _grid_auc_quantiles(fit, 1350.0)
    normal = np.random.default_rng(0).multivariate_normal(fit.eta, fit.eta_covariance, 4000)
    auc = 1350.0 / np.array([fit.model.individual(fit.patient, e)["CL"] for e in normal])
    assert np.percentile(auc, 5) < 0.9 * lo
    assert vd.predict_exposure(fit, 450.0, 8.0, 1.0).auc24_ci[0] > 0.97 * lo


def test_prior_only_fits_use_the_exact_normal_prior():
    resp = vd.evaluate(_request(levels=[], options={"model": "smit2021", "series": False}))
    e = resp["results"][0]
    assert e["posterior_method"] == "prior" and e["posterior_ess"] is None


def test_posterior_draws_are_reproducible():
    a = vd.evaluate(_request(options={"model": "smit2021", "series": False}))
    b = vd.evaluate(_request(options={"model": "smit2021", "series": False}))
    assert a["results"][0]["current"]["auc24_ci"] == b["results"][0]["current"]["auc24_ci"]
    assert a["results"][0]["posterior_method"] == "importance sampling"
    assert a["results"][0]["posterior_ess"] > 1000


def test_unreliable_weights_fall_back_to_the_normal_approximation(monkeypatch):
    from vankedoodle import bayes
    monkeypatch.setattr(bayes, "MIN_EFFECTIVE_DRAWS", 10 ** 9)
    fit = _single_trough_fit()
    exp = vd.predict_exposure(fit, 450.0, 8.0, 1.0)
    assert fit.posterior_method == "normal approximation"
    assert exp.auc24_ci is not None
    assert any("could not be sampled reliably" in w for w in fit.warnings)


def test_objective_is_finite_far_from_the_mode():
    fit = _single_trough_fit()
    for eta in ([8.0, -8.0], [-8.0, 8.0], [20.0, 20.0]):
        assert np.isfinite(fit._objective(np.array(eta)))


def test_history_curves_run_until_the_next_dose_is_due():
    s = vd.evaluate(_request(options={"model": "smit2021", "series": True}))["results"][0]["series"]
    # last dose at 32 h, every 8 h: the curves carry on to 40 h
    assert s["time_h"][-1] == pytest.approx(40.0)
