"""Headless tests for the Streamlit interface.

Streamlit's AppTest runs app.py in-process without a browser, so the page is
exercised for real: widgets are set, the script re-runs, and the rendered
elements are inspected. The chart builders are tested directly on a response.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

import vankedoodle as vd

pytest.importorskip("plotly")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
EXAMPLE = ROOT / "example_case.json"
TIMEOUT = 180


@pytest.fixture(scope="module")
def response():
    req = json.loads(EXAMPLE.read_text(encoding="utf8"))
    req["options"].update({"model": "smit2021", "uncertainty": True,
                           "continuous_infusion": True, "peak_threshold_mg_l": 50.0})
    return vd.evaluate(req)


@pytest.fixture(scope="module")
def comparison():
    return vd.evaluate({**json.loads(EXAMPLE.read_text(encoding="utf8")),
                        "options": {"model": "all", "series": False}})


@pytest.fixture(scope="module")
def comparison_with_series():
    return vd.evaluate({**json.loads(EXAMPLE.read_text(encoding="utf8")),
                        "options": {"model": "all", "series": True}})


# ---------------------------------------------------------------- charts
def _names(fig):
    return {t.name for t in fig.data if t.name}


def test_concentration_figure_has_every_layer_by_default(response):
    import app_charts
    fig = app_charts.concentration_figure(response, "smit2021")
    fig.to_dict()
    assert {"Population prior", "Fitted (smit2021)", "Forecasted concentration",
            "Prior 95% band", "Fitted 95% band", "Forecast 95% band",
            "Measured levels"} <= _names(fig)
    assert not any("Dose" in n for n in _names(fig))
    assert not fig.layout.title.text
    levels = next(t for t in fig.data if t.name == "Measured levels")
    assert len(levels.x) == len(response["levels"])


def test_bands_share_their_curves_legend_entry(response):
    import app_charts
    fig = app_charts.concentration_figure(response)
    shown = [t.name for t in fig.data if t.showlegend is not False and t.name]
    assert not any("band" in n for n in shown)
    assert len(shown) <= 4                                  # one row at most widths
    band = next(t for t in fig.data if t.name == "Fitted 95% band")
    line = next(t for t in fig.data if t.name == "Fitted (smit2021)")
    assert band.legendgroup == line.legendgroup


def test_other_models_can_be_overlaid(response, comparison_with_series):
    import app_charts
    plain = app_charts.concentration_figure(response)
    assert not any(n.startswith("Fitted (") and "smit2021" not in n for n in _names(plain))
    fig = app_charts.concentration_figure(response, overlay=comparison_with_series,
                                          overlay_models=["le2013", "xu2026"])
    fig.to_dict()
    added = [t for t in fig.data if t.name in ("Fitted (le2013)", "Fitted (xu2026)")]
    assert len(added) == 2
    assert all(t.line.dash == "dot" for t in added)
    assert added[0].line.color != added[1].line.color


def test_forecast_can_come_from_another_model(response):
    import app_charts
    req = json.loads(EXAMPLE.read_text(encoding="utf8"))
    req["options"].update({"model": "xu2026", "uncertainty": True})
    xu = vd.evaluate(req)["results"][0]
    fig = app_charts.concentration_figure(response, forecast_from=xu,
                                          model_order=sorted(vd.MODELS))
    names = _names(fig)
    assert "Forecasted concentration (xu2026)" in names
    assert "Fitted (xu2026)" in names                 # the fit the forecast continues
    assert "Fitted (smit2021)" in names
    fc = next(t for t in fig.data if t.name == "Forecasted concentration (xu2026)")
    assert list(fc.y) != list(next(t for t in app_charts.concentration_figure(
        response).data if t.name == "Forecasted concentration").y)


def test_layers_can_be_switched_off(response):
    import app_charts
    off = {k: False for k in app_charts.DEFAULT_LAYERS}
    fig = app_charts.concentration_figure(response, layers={**off, "fitted": True})
    assert _names(fig) == {"Fitted (smit2021)"}


def test_forecast_regimen_is_written_at_the_bottom(response):
    import app_charts
    fig = app_charts.concentration_figure(response, peak_line=50.0)
    labels = [a for a in fig.layout.annotations if "q8hr" in a.text]
    assert len(labels) == 1
    assert "525mg q8hr" in labels[0].text                  # the example's candidate
    assert labels[0].yref == "paper" and labels[0].y == 0.0     # bottom of the plot
    assert labels[0].font.color == app_charts.PROJECTION_COLOUR
    assert fig.layout.yaxis.range[1] > 50.0                # room for the label
    off = app_charts.concentration_figure(response, peak_line=50.0,
                                          layers={"forecast": False})
    assert not [a for a in off.layout.annotations if "q8hr" in a.text]


def test_current_regimen_is_written_in_green_and_peak_label_sits_below(response):
    import app_charts
    fig = app_charts.concentration_figure(response, peak_line=50.0)
    given = [a for a in fig.layout.annotations if "450mg q8hr" in a.text]
    assert len(given) == 1
    assert given[0].font.color == app_charts.POSTERIOR_COLOUR
    assert given[0].yref == "paper" and given[0].y == 0.0
    assert pd.Timestamp(given[0].x) == pd.Timestamp("2026-03-02 08:00")   # first dose
    peak = [a for a in fig.layout.annotations if "peak 50" in a.text]
    assert peak and peak[0].yanchor == "top"                             # below the line


def test_legends_sit_above_their_plots(response, comparison):
    """Below the plot a wrapping legend collides with the axis labels."""
    import app_charts
    figs = [app_charts.concentration_figure(response), app_charts.compartment_figure(response)]
    for fig in figs:
        assert fig.layout.legend.y >= 1 and fig.layout.legend.yanchor == "bottom"
        assert fig.layout.margin.t >= app_charts.LEGEND_SPACE_PX      # room reserved
        assert fig.layout.legend.xanchor == "left"
        assert not fig.layout.title.text


def test_sidebar_shows_doses_and_levels_as_tables(page):
    tables = {tuple(d.value.columns): d.value for d in page.sidebar.dataframe}
    doses = tables[("#", "Given", "Dose (mg)", "Infusion (h)")]
    levels = tables[("#", "Drawn", "Level (mg/L)")]
    assert len(doses) == 5 and list(doses["Dose (mg)"])[0] == "450"
    assert list(levels["Level (mg/L)"]) == ["6.2", "30.2", "8.3"]
    headers = [h.value for h in page.sidebar.header]
    assert "Vancomycin Dosing Information" in headers
    assert "Vancomycin Blood Levels" in headers


def test_alternate_dosing_table_has_five_rows(page):
    assert any("Alternate Dosing Recommendations" in h.value for h in page.subheader)
    grid = next(d.value for d in page.main.dataframe if "Regimen" in d.value.columns)
    assert list(grid["Regimen"]) == ["q6hr", "q8hr", "q12hr", "q24hr",
                                     "Continuous infusion"]


def test_regimen_summaries(response):
    import app_charts
    assert app_charts.given_regimen_summary(response) == "450 mg every 8 h × 5 doses"
    assert app_charts.forecast_regimen_summary(response).startswith(
        "525 mg every 8 h")                                    # the example's candidate


def test_forecast_starts_where_the_fitted_curve_needs_it(response):
    import app_charts
    entry = app_charts.result_for(response)
    f = entry["forecast"]
    assert f["next_dose_h"] == 40.0                    # last dose at 32 h, every 8 h
    fig = app_charts.concentration_figure(response)
    fc = next(t for t in fig.data if t.name == "Forecasted concentration")
    assert len(fc.x) > 100
    # the forecast starts at the first forecast dose, where the fitted curve ends
    fitted = next(t for t in fig.data if t.name == "Fitted (smit2021)")
    assert pd.Timestamp(fc.x[0]) == pd.Timestamp("2026-03-02 08:00") + pd.Timedelta(hours=40)
    gap = abs(pd.Timestamp(fitted.x[-1]) - pd.Timestamp(fc.x[0]))
    assert gap < pd.Timedelta(seconds=1)


def test_auc_readout(response):
    import app_charts
    entry = app_charts.result_for(response)
    html = app_charts.auc_grid_html([("Current", entry["current"]),
                                     ("Forecasted", entry["candidate"])], response["target"])
    assert "grid-template-columns" in html                   # two columns
    assert "Current AUC<sub>24</sub>" in html and "Forecasted AUC<sub>24</sub>" in html
    assert html.count("90% CI") == 2
    assert f"{entry['current']['auc24']:.0f}" in html


def test_compartment_figure_needs_the_two_level_result(response):
    import app_charts
    fig = app_charts.compartment_figure(response)
    fig.to_dict()
    assert {"One-compartment (two-level)", "smit2021, day-3 interval",
            "Levels used"} <= _names(fig)
    used = next(t for t in fig.data if t.name == "Levels used")
    assert len(used.x) == 2                  # only the two levels the method used
    one = vd.evaluate({**json.loads(EXAMPLE.read_text(encoding="utf8")),
                       "levels": [{"time": "2026-03-03 07:30", "conc_mg_l": 6.2}]})
    assert app_charts.compartment_figure(one) is None


def test_comparison_table_recommends_a_daily_dose(comparison):
    import app_charts
    table = app_charts.comparison_table(comparison)
    assert list(table.columns) == ["Model", "CL (L/h)", "AUC24 now", "Recommended"]
    assert table["Recommended"].str.endswith("mg/day").all()


def test_model_comparison_figure_shows_every_model(comparison):
    import app_charts
    fig = app_charts.model_comparison_figure(comparison, x_range=(200, 800))
    fig.to_dict()
    assert list(fig.layout.xaxis.range) == [200, 800]
    assert {y.rstrip(" *") for y in fig.data[0].y} == set(vd.MODELS)


def test_two_level_table(response):
    import app_charts
    table = app_charts.two_level_table(response)
    assert list(table.columns) == ["", "Two-level", "Bayesian (smit2021)", "Difference"]
    sz, entry = response["sawchuk_zaske"], app_charts.result_for(response)
    expected = 100 * (entry["current"]["auc24"] - sz["auc24"]) / sz["auc24"]
    assert table["Difference"].iloc[0] == f"{expected:+.1f}%"
    assert "L/kg" in table["Two-level"].iloc[3]


def test_tables(response):
    import app_charts
    entry = app_charts.result_for(response)
    assert list(app_charts.parameter_table(entry)["Parameter"])[0] == "Clearance (L/h)"
    cv = app_charts.uncertainty_table(entry)
    assert (cv["Post-fit CV (%)"] < cv["Prior CV (%)"]).all()
    options = app_charts.regimen_options(entry)
    assert [o["label"] for o in options] == ["q6hr", "q8hr", "q12hr", "q24hr",
                                             "Continuous infusion"]
    table = app_charts.regimen_table(options, response["target"])
    assert list(table.columns) == ["Regimen", "Optimal dose", "%PTA", "AUC < 400 Risk",
                                   "AUC > 600 Risk", "Peak > 50 mg/L Risk"]
    assert table["%PTA"].str.endswith("%").all()             # text, not a bar
    assert table["Optimal dose"].iloc[-1].endswith("mg/24 hr")


def test_charts_refuse_a_response_without_series(response):
    import app_charts
    bare = vd.evaluate({**json.loads(EXAMPLE.read_text(encoding="utf8")),
                        "options": {"model": "smit2021", "series": False}})
    with pytest.raises(ValueError, match="series"):
        app_charts.concentration_figure(bare)


# ---------------------------------------------------------------- the page
@pytest.fixture
def page():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_app_runs_clean_with_its_defaults(page):
    assert not page.exception
    assert not page.sidebar.title                    # the logo replaces the text title
    html = " ".join(m.value for m in page.markdown)
    assert "Current AUC<sub>24</sub>" in html
    assert "Forecasted AUC<sub>24</sub>" in html
    assert "AUC24, current regimen" not in [m.label for m in page.metric]
    assert any("Alternate Dosing Recommendations" in h.value for h in page.subheader)
    assert not any("Predictive regimen evaluation" in h.value for h in page.subheader)


def test_case_inputs_are_in_the_sidebar_and_result_controls_beside_results(page):
    for key in ("weight_kg", "target_auc", "peak_threshold"):
        assert page.sidebar.number_input(key=key) is not None
    assert page.sidebar.date_input(key="first_date") is not None
    assert page.sidebar.number_input(key="peak_threshold").value == 50.0
    assert not [r for r in page.sidebar.radio if r.key == "time_mode"]
    main_keys = {w.key for w in list(page.main.number_input) + list(page.main.checkbox)}
    assert {"cand_dose", "cand_interval", "cand_infusion", "show_bands",
            "show_trough"} <= main_keys
    assert page.sidebar.slider(key="trough_range").value == (8.0, 12.0)


def _patient_table(page):
    return next(d.value for d in page.sidebar.dataframe
                if list(d.value.columns) == ["Field", "Value"])


def test_patient_table_shows_neonatal_fields_only_under_one_year(page):
    fields = list(_patient_table(page)["Field"])
    assert fields[:2] == ["Birthdate", "Age"]
    assert "Gestational age" not in fields
    page.sidebar.date_input(key="birthdate").set_value(
        dt.date.today() - dt.timedelta(days=45)).run()
    assert not page.exception
    table = _patient_table(page)
    assert "Gestational age" in set(table["Field"])
    assert table.set_index("Field").loc["Age", "Value"] == "45 days"
    assert page.sidebar.number_input(key="gest_age") is not None


def test_app_reports_the_two_level_result_with_the_default_levels(page):
    table = next(d.value for d in page.main.dataframe if "Two-level" in d.value.columns)
    assert list(table[""]) == ["AUC24 (mg·h/L)", "Clearance (L/h)", "Half-life (h)",
                               "Volume of distribution"]
    assert table["Difference"].iloc[0].endswith("%")
    assert "Peak risk" in labels


def test_app_explains_itself_when_only_one_level_is_entered(page):
    page.session_state["level_rows"] = [{"time_h": 23.5, "conc_mg_l": 6.2}]
    page.session_state["table_version"] = 1
    page.run()
    assert not page.exception
    assert any("Two levels are needed" in i.value for i in page.info)


def test_app_surfaces_out_of_population_warnings(page):
    """The default case has CLcr above the Smit cap, which must be visible."""
    assert any("extrapolation" in w.value for w in page.warning)


def test_app_compare_all_adds_the_model_table(page):
    page.sidebar.checkbox(key="compare_all").set_value(True).run()
    assert not page.exception
    assert any("How the models compare" in h.value for h in page.subheader)
    tables = [d for d in page.dataframe if "Model" in list(d.value.columns)]
    assert len(tables) == 1
    assert set(tables[0].value["Model"]) == set(vd.MODELS)
    assert list(tables[0].value.columns) == ["Model", "CL (L/h)", "AUC24 now", "Recommended"]


def test_app_test_regimen_is_forecast(page):
    page.number_input(key="cand_dose").set_value(600.0)
    page.number_input(key="cand_interval").set_value(12.0).run()
    assert not page.exception
    assert any("Forecasted AUC<sub>24</sub>" in m.value for m in page.markdown)
    assert not [b for b in page.button if b.label == "Use recommended"]


def test_overlay_checkbox_adds_another_models_fit(page):
    page.checkbox(key="overlay_le2013").check().run()
    assert not page.exception


def test_dosing_model_changes_the_table_and_follows_the_sidebar(page):
    first = next(d.value for d in page.main.dataframe if "Regimen" in d.value.columns)
    page.selectbox(key="dosing_model").set_value("xu2026").run()
    assert not page.exception
    second = next(d.value for d in page.main.dataframe if "Regimen" in d.value.columns)
    assert list(first["Optimal dose"]) != list(second["Optimal dose"])
    assert any("forecast use **xu2026**" in i.value for i in page.info)
    page.sidebar.selectbox(key="model").set_value("le2013").run()
    assert page.selectbox(key="dosing_model").value == "le2013"


def test_page_forecasts_48_hours():
    import ast
    tree = ast.parse(Path(APP).read_text(encoding="utf8"))
    horizon = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", None) == "FORECAST_H")
    assert horizon == 48.0
    req = json.loads(EXAMPLE.read_text(encoding="utf8"))
    req["options"].update({"model": "smit2021", "uncertainty": True,
                           "forecast_h": horizon})
    f = vd.evaluate(req)["results"][0]["forecast"]
    assert f["time_h"][-1] == pytest.approx(f["forecast_start_h"] + 48.0)


def test_app_shows_errors_instead_of_crashing_on_bad_input(page):
    page.sidebar.number_input(key="height_cm").set_value(20.0)
    page.sidebar.number_input(key="scr_value").set_value(0.01)
    page.run()
    assert not page.exception                          # never a traceback on screen


def test_logo_is_outlined_and_transparent():
    """The logo carries no text (so needs no fonts) and no background of its own,
    so it sits on whatever the theme's background is."""
    svg = (ROOT / "assets" / "vankedoodle_logo.svg").read_text(encoding="utf8")
    assert svg.startswith("<svg") and "viewBox" in svg
    assert "<text" not in svg and "<rect" not in svg and "<image" not in svg
    assert "#00D2FF" in svg                       # the theme's primary colour
    assert (ROOT / "assets" / "vankedoodle_icon.png").exists()
