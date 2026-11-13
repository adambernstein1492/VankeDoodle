"""Chart builders for the VankeDoodle interface.

Pure functions: each takes the JSON response from ``vankedoodle.evaluate`` and
returns an Altair chart. Nothing here imports Streamlit, so the charts can be
tested headlessly and reused by any other front end.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import altair as alt
import pandas as pd

PRIOR_COLOUR = "#9aa5b1"
POSTERIOR_COLOUR = "#2b6cb0"
PROJECTION_COLOUR = "#2f855a"
LEVEL_COLOUR = "#c0392b"
TARGET_COLOUR = "#8fbf8f"


def _result_for(response: Mapping, model: Optional[str] = None) -> Dict[str, Any]:
    usable = [r for r in response["results"] if r.get("ok")]
    if not usable:
        raise ValueError("No model produced a result for this case.")
    return next((r for r in usable if r["model"] == model), usable[0])


def concentration_chart(response: Mapping, model: Optional[str] = None,
                        height: int = 300) -> alt.LayerChart:
    """Concentration against time: prior, posterior, measured levels, projection."""
    entry = _result_for(response, model)
    series = entry.get("series")
    if not series:
        raise ValueError("The response carries no series; request options.series = true.")

    frames = [
        pd.DataFrame({"time_h": series["time_h"], "conc": series["conc_prior"],
                      "curve": "population prior"}),
        pd.DataFrame({"time_h": series["time_h"], "conc": series["conc_posterior"],
                      "curve": f"fitted ({entry['model']})"}),
    ]
    if series.get("projection_time_h"):
        frames.append(pd.DataFrame({"time_h": series["projection_time_h"],
                                    "conc": series["projection_conc"],
                                    "curve": "recommended regimen"}))
    curves = pd.concat(frames, ignore_index=True)
    order = [c for c in ("population prior", f"fitted ({entry['model']})",
                         "recommended regimen") if c in set(curves["curve"])]
    colours = {"population prior": PRIOR_COLOUR,
               f"fitted ({entry['model']})": POSTERIOR_COLOUR,
               "recommended regimen": PROJECTION_COLOUR}

    lines = alt.Chart(curves).mark_line().encode(
        x=alt.X("time_h:Q", title="Hours after the first dose"),
        y=alt.Y("conc:Q", title="Vancomycin (mg/L)"),
        color=alt.Color("curve:N", title=None,
                        scale=alt.Scale(domain=order,
                                        range=[colours[c] for c in order]),
                        legend=alt.Legend(orient="top-left")),
        strokeDash=alt.StrokeDash("curve:N", legend=None,
                                  scale=alt.Scale(domain=order,
                                                  range=[[1, 0], [1, 0], [5, 3]][:len(order)])),
        tooltip=[alt.Tooltip("time_h:Q", title="h", format=".1f"),
                 alt.Tooltip("conc:Q", title="mg/L", format=".1f"),
                 alt.Tooltip("curve:N", title="")],
    )

    layers: List[alt.Chart] = [lines]
    if response["levels"]:
        points = pd.DataFrame(response["levels"])
        layers.append(
            alt.Chart(points).mark_point(size=90, filled=True, color=LEVEL_COLOUR)
            .encode(x="time_h:Q", y="conc_mg_l:Q",
                    tooltip=[alt.Tooltip("time_h:Q", title="h", format=".2f"),
                             alt.Tooltip("conc_mg_l:Q", title="measured mg/L",
                                         format=".1f")]))
    return alt.layer(*layers).properties(height=height).interactive()


def auc_chart(response: Mapping, height: int = 220) -> alt.LayerChart:
    """Estimated AUC24 per model against the target window, with the two-level value."""
    rows = []
    for entry in response["results"]:
        if not entry.get("ok"):
            continue
        current = entry["current"]
        ci = current.get("auc24_ci") or [None, None]
        rows.append({"model": entry["model"] + (" *" if entry["out_of_population"] else ""),
                     "auc24": current["auc24"], "low": ci[0], "high": ci[1],
                     "in_target": current["in_target"]})
    if not rows:
        raise ValueError("No model produced a result for this case.")
    data = pd.DataFrame(rows)
    target = response["target"]

    band = alt.Chart(pd.DataFrame({"low": [target["low"]], "high": [target["high"]]})
                     ).mark_rect(opacity=0.18, color=TARGET_COLOUR).encode(
        x="low:Q", x2="high:Q")
    # start the axis at the data rather than at zero: AUC24 values of interest sit
    # in a narrow band and a zero-anchored axis wastes most of the width
    intervals = alt.Chart(data).mark_rule(size=2, color=POSTERIOR_COLOUR).encode(
        x=alt.X("low:Q", title="Estimated AUC24 on the current regimen (mg·h/L)",
                scale=alt.Scale(zero=False, nice=True, padding=12)),
        x2="high:Q", y=alt.Y("model:N", title=None, sort=None))
    points = alt.Chart(data).mark_point(size=90, filled=True,
                                        color=POSTERIOR_COLOUR).encode(
        x="auc24:Q", y=alt.Y("model:N", sort=None),
        tooltip=[alt.Tooltip("model:N", title=""),
                 alt.Tooltip("auc24:Q", title="AUC24", format=".0f"),
                 alt.Tooltip("low:Q", title="90% low", format=".0f"),
                 alt.Tooltip("high:Q", title="90% high", format=".0f")])
    layers = [band, intervals, points]

    sz = response["sawchuk_zaske"]
    if sz.get("available"):
        layers.append(
            alt.Chart(pd.DataFrame({"auc24": [sz["auc24"]]}))
            .mark_rule(color=LEVEL_COLOUR, strokeDash=[5, 3], size=2)
            .encode(x="auc24:Q",
                    tooltip=[alt.Tooltip("auc24:Q", title="two-level AUC24",
                                         format=".0f")]))
    return alt.layer(*layers).properties(height=height)
