"""Headless tests for the Streamlit interface.

Streamlit's AppTest runs app.py in-process without a browser, so the page is
exercised for real: widgets are set, the script re-runs, and the rendered
elements are inspected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import vankedoodle as vd

alt = pytest.importorskip("altair")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
EXAMPLE = ROOT / "example_case.json"
TIMEOUT = 120


@pytest.fixture(scope="module")
def response():
    return vd.evaluate(json.loads(EXAMPLE.read_text(encoding="utf8")))


# ---------------------------------------------------------------- charts
def test_concentration_chart_has_every_curve_and_the_levels(response):
    import app_charts
    chart = app_charts.concentration_chart(response, "smit2021")
    chart.to_dict()                                     # the spec must be valid
    assert len(chart.layer) == 2                        # curves + measured points
    curves = set(chart.layer[0].data["curve"])
    assert curves == {"population prior", "fitted (smit2021)", "recommended regimen"}
    assert len(chart.layer[1].data) == len(response["levels"])


def test_auc_chart_shows_every_model_and_the_two_level_value(response):
    import app_charts
    chart = app_charts.auc_chart(response)
    chart.to_dict()                                     # the spec must be valid
    models = {m.rstrip(" *") for m in chart.layer[1].data["model"]}
    assert models == {r["model"] for r in response["results"] if r["ok"]}
    # band, intervals, points, and the Sawchuk-Zaske rule
    assert len(chart.layer) == 4


def test_charts_refuse_a_response_without_series(response):
    import app_charts
    bare = vd.evaluate({**json.loads(EXAMPLE.read_text(encoding="utf8")),
                        "options": {"model": "smit2021", "series": False}})
    with pytest.raises(ValueError, match="series"):
        app_charts.concentration_chart(bare)


# ---------------------------------------------------------------- the page
def test_app_runs_clean_with_its_defaults():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "VankeDoodle"
    assert len(at.metric) >= 8                      # current state and recommendation
    assert any("Recommended next regimen" in h.value for h in at.subheader)


def test_app_starts_with_no_levels_entered():
    """Pre-filled measurements are unsafe: they survive a dose edit and fit a
    patient who does not exist. The page must start with an empty levels table."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("No levels entered" in c.value for c in at.caption)
    assert any("Two levels are needed" in i.value for i in at.info)
    # the population prediction is still shown
    assert any(m.label == "AUC24, current regimen" for m in at.metric)


def test_app_loads_the_example_case_only_on_request():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    button = next(b for b in at.button if "example case" in b.label)
    button.click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "AUC24, two-level" in labels
    assert "Bayesian vs two-level" in labels
    assert any("Fitting 3 measured level" in c.value for c in at.caption)


def test_app_warns_when_levels_cannot_match_the_doses():
    """The reported failure: example levels left behind after the dose is changed.

    15 mg q8h cannot produce a 30 mg/L concentration; the page must say so rather
    than present the resulting AUC as a finding.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    next(b for b in at.button if "example case" in b.label).click().run()
    at.number_input(key="dose_mg_input").set_value(15.0).run()
    assert not at.exception
    assert any("cannot be reconciled" in w.value for w in at.warning)


def test_app_surfaces_out_of_population_warnings():
    """The default case has CLcr above the Smit cap, which must be visible."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("extrapolation" in w.value for w in at.warning)


def test_app_compare_all_adds_the_model_table():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.checkbox[0].set_value(True).run()
    assert not at.exception
    assert any("How the models compare" in h.value for h in at.subheader)
    tables = [d for d in at.dataframe if "model" in list(d.value.columns)]
    assert len(tables) == 1
    assert set(tables[0].value["model"]) == set(vd.MODELS)


def test_app_candidate_regimen_is_reported():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.number_input(key="cand_dose").set_value(600.0).run()
    assert not at.exception
    assert any("Your regimen" in c.value for c in at.caption)


def test_app_shows_errors_instead_of_crashing_on_bad_input():
    """Zero height makes bedside Schwartz impossible for the default model."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.number_input[2].set_value(20.0)      # height, minimum allowed
    at.sidebar.number_input[3].set_value(0.01)      # creatinine -> huge CLcr
    at.run()
    assert not at.exception                          # never a traceback on screen
