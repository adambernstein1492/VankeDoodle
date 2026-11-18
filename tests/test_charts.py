"""The main concentration plot."""
from __future__ import annotations

import pytest

import vankedoodle as vd

go = pytest.importorskip("plotly.graph_objects")
if not hasattr(go.Figure().layout, "xaxis"):          # a stand-in, not real Plotly
    pytest.skip("needs the real plotly", allow_module_level=True)

import app_charts  # noqa: E402


def _response(n_doses: int):
    req = {"patient": {"weight_kg": 23.0, "age_years": 7.0, "height_cm": 122.0,
                       "scr_mg_dl": 0.33},
           "regimen": {"dose_mg": 450, "interval_h": 8, "n_doses": n_doses, "start": 0.0},
           "levels": [{"time": 8.0 * (n_doses - 2) + 2.0, "conc_mg_l": 30.0}],
           "options": {"model": "smit2021", "uncertainty": True, "forecast_h": 48.0,
                       "series": True}}
    resp = vd.evaluate(req)
    return resp, resp["results"][0]


def test_short_history_shows_everything():
    resp, entry = _response(5)
    fig = app_charts.concentration_figure(resp, entry["model"])
    assert fig.layout.xaxis.range is None
    assert not fig.layout.xaxis.rangeslider.visible


def test_long_history_opens_on_the_last_three_days_and_scrolls():
    resp, entry = _response(30)                       # 10 days of q8h
    fig = app_charts.concentration_figure(resp, entry["model"])
    lo, hi = fig.layout.xaxis.range
    x = app_charts._clock(resp)
    f = entry["forecast"]
    assert lo == x([f["next_dose_h"] - 72.0])[0]
    assert hi == x([f["time_h"][-1]])[0]
    assert fig.layout.xaxis.rangeslider.visible
    assert fig.layout.dragmode == "pan"
    assert any((tr.name or "").startswith("Forecasted") for tr in fig.data)   # bands are unnamed
