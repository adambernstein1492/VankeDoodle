"""Chart and table builders for the VankeDoodle interface.

Pure functions: each takes the JSON response from ``vankedoodle.evaluate`` (or
one model's entry from it) and returns a Plotly figure or a pandas DataFrame.
Nothing here imports Streamlit or computes a clinical quantity, so the builders
can be tested headlessly and reused by any other front end.

The look follows a dark Streamlit theme (see .streamlit/config.toml): transparent
backgrounds, faint grid lines, and one colour per meaning throughout the page.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

PRIOR_COLOUR = "#95a5a6"
POSTERIOR_COLOUR = "#2ecc71"
PROJECTION_COLOUR = "#00d2ff"
LEVEL_COLOUR = "#e74c3c"
MEAN_COLOUR = "#3498db"
ONE_CMT_COLOUR = "#4b8bdf"
BELOW_COLOUR = "#f1c40f"
TARGET_COLOUR = "#2ecc71"
ABOVE_COLOUR = "#e74c3c"
GRID_COLOUR = "rgba(255, 255, 255, 0.10)"
LEGEND_SPACE_PX = 56
# Other models' fits, when overlaid: one colour per model, by its place in the
# response, so a model keeps its colour as others are switched on and off.
# None is green or blue, which are taken by the main fit and the forecast.
OVERLAY_COLOURS = ["#f39c12", "#9b59b6", "#e84393", "#fdcb6e", "#a29bfe", "#e17055",
                   "#d4a373"]        # strip kept above every plot for its legend


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _style(fig: go.Figure, x_title: str, y_title: str,
           title: Optional[str] = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=legend,
        # The legend sits in a strip reserved above the plot. Plotly can grow the
        # margin to fit a legend by itself, but not reliably inside Streamlit, so
        # the room is set aside explicitly (and may still grow if the legend wraps).
        margin=dict(l=0, r=0, b=0, autoexpand=True,
                    t=(LEGEND_SPACE_PX if legend else 10) + (30 if title else 0)),
        # Above the plot, left-aligned: the top margin grows to fit however many
        # rows the legend wraps to, and nothing else (axis labels, the toolbar at
        # top right) competes for that space, so it cannot overlap the plot.
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOUR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOUR, zeroline=False, rangemode="tozero")
    return fig


def _reference_lines(fig: go.Figure, trough_band: Optional[Tuple[float, float]],
                     peak_line: Optional[float]) -> None:
    if trough_band:
        fig.add_hrect(y0=trough_band[0], y1=trough_band[1], fillcolor="gray",
                      opacity=0.15, layer="below", line_width=0)
    if peak_line:
        fig.add_hline(y=peak_line, line_dash="dash", line_color=LEVEL_COLOUR,
                      layer="below", annotation_text=f"peak {peak_line:g} mg/L",
                      annotation_position="bottom left",
                      annotation_font=dict(size=11, color=LEVEL_COLOUR))


def _band(fig: go.Figure, x: Sequence[float], band: Optional[Mapping], colour: str,
          name: str, group: str) -> None:
    if not band or band.get("low") is None:
        return
    fig.add_trace(go.Scatter(x=list(x), y=band["high"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", legendgroup=group))
    # No legend entry of its own: the band shares its curve's legend group, so it
    # hides and shows with that curve, and the legend stays short enough for one row.
    fig.add_trace(go.Scatter(x=list(x), y=band["low"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=_rgba(colour, 0.20),
                             name=name, showlegend=False, hoverinfo="skip",
                             legendgroup=group))


def result_for(response: Mapping, model: Optional[str] = None) -> Dict[str, Any]:
    """The entry for ``model``, or the first model that produced a result."""
    usable = [r for r in response["results"] if r.get("ok")]
    if not usable:
        raise ValueError("No model produced a result for this case.")
    return next((r for r in usable if r["model"] == model), usable[0])


# ======================================================================
# Concentration over time: history, fit and forecast on one plot
# ======================================================================
DEFAULT_LAYERS = {
    "prior": True, "prior_band": True,
    "fitted": True, "fitted_band": True,
    "forecast": True, "forecast_band": True,
    "levels": True,
}


def _clock(response: Mapping):
    """A function mapping hours after the first dose to clock time, when known."""
    origin = (response.get("regimen_given") or {}).get("time_origin")
    if not origin:
        return lambda hours: list(hours)
    t0 = pd.Timestamp(origin)
    return lambda hours: [t0 + pd.Timedelta(hours=float(h)) for h in hours]


def _when(response: Mapping, hours: float) -> str:
    origin = (response.get("regimen_given") or {}).get("time_origin")
    if not origin:
        return f"hour {hours:g}"
    return (pd.Timestamp(origin) + pd.Timedelta(hours=float(hours))).strftime("%b %d %H:%M")


def given_regimen_summary(response: Mapping) -> str:
    """The doses actually given, in one phrase."""
    doses = sorted(response["regimen_given"]["doses"], key=lambda d: d["time_h"])
    amounts = {round(d["amount_mg"], 3) for d in doses}
    gaps = {round(b["time_h"] - a["time_h"], 3) for a, b in zip(doses, doses[1:])}
    n = len(doses)
    if len(amounts) == 1 and len(gaps) <= 1:
        every = f" every {gaps.pop():g} h" if gaps else ""
        return f"{doses[0]['amount_mg']:.0f} mg{every} × {n} dose{'s' if n > 1 else ''}"
    return f"{n} doses, {min(amounts):.0f}-{max(amounts):.0f} mg"


def given_label(response: Mapping) -> str:
    """The current regimen in shorthand, e.g. "450mg q8hr": the last dose given
    and the interval it was given at."""
    g = response["regimen_given"]
    return f"{g['last_dose_mg']:.0f}mg q{g['observed_interval_h']:g}hr"


def forecast_label(forecast: Mapping) -> str:
    """The forecast regimen in shorthand, e.g. "325mg q6hr", or
    "1300mg/24hr continuous" for an infusion that fills its interval."""
    if forecast["infusion_h"] >= forecast["interval_h"]:
        daily = forecast["dose_mg"] * 24.0 / forecast["interval_h"]
        return f"{daily:.0f}mg/24hr continuous"
    return f"{forecast['dose_mg']:.0f}mg q{forecast['interval_h']:g}hr"


def forecast_regimen_summary(response: Mapping, model: Optional[str] = None) -> str:
    f = result_for(response, model)["forecast"]
    return (f"{f['dose_mg']:.0f} mg every {f['interval_h']:g} h over {f['infusion_h']:g} h, "
            f"from {_when(response, f['next_dose_h'])}")


def overlay_colour(order: Sequence[str], model: str) -> str:
    """A model's overlay colour, from its place in ``order`` (all model names)."""
    i = list(order).index(model) if model in order else len(order)
    return OVERLAY_COLOURS[i % len(OVERLAY_COLOURS)]


def concentration_figure(response: Mapping, model: Optional[str] = None,
                         layers: Optional[Mapping[str, bool]] = None,
                         trough_band: Optional[Tuple[float, float]] = None,
                         peak_line: Optional[float] = None,
                         overlay: Optional[Mapping] = None,
                         overlay_models: Sequence[str] = (),
                         forecast_from: Optional[Mapping] = None,
                         model_order: Optional[Sequence[str]] = None) -> go.Figure:
    """The whole course on one plot: population prior, fitted curve and measured
    levels over the doses given, then the forecasted concentration under the
    regimen being evaluated, each with an optional 95% band.

    ``layers`` switches parts on and off (keys as in ``DEFAULT_LAYERS``).
    ``overlay`` is a response for ``model="all"`` with series; the fitted curve of
    each model named in ``overlay_models`` is drawn over the others, dotted.
    Models that produced no fit are skipped.

    ``forecast_from`` is another model's entry (with ``forecast`` and ``series``)
    whose forecast replaces this model's: its fitted curve is then drawn dotted
    too, so the forecast continues from the fit it came from. ``model_order``
    (every model name) keeps each model's overlay colour fixed.
    """
    show = {**DEFAULT_LAYERS, **(layers or {})}
    entry = result_for(response, model)
    s = entry.get("series")
    if not s:
        raise ValueError("The response carries no series; request options.series = true.")
    other = forecast_from if forecast_from and forecast_from["model"] != entry["model"] else None
    f = (other or entry).get("forecast")
    order = list(model_order or ([r["model"] for r in overlay["results"]] if overlay else [])
                 or ([other["model"]] if other else []))
    x = _clock(response)

    fig = go.Figure()
    if f and show["forecast"]:
        fig.add_vrect(x0=x([f["next_dose_h"]])[0], x1=x([f["time_h"][-1]])[0],
                      fillcolor="rgba(200, 200, 200, 0.10)", layer="below", line_width=0)
    _reference_lines(fig, trough_band, peak_line)

    t = x(s["time_h"])
    if show["prior"]:
        if show["prior_band"]:
            _band(fig, t, s.get("band_prior"), PRIOR_COLOUR, "Prior 95% band", "prior")
        fig.add_trace(go.Scatter(x=t, y=s["conc_prior"], mode="lines",
                                 line=dict(color=PRIOR_COLOUR, width=2),
                                 name="Population prior", legendgroup="prior"))
    if show["fitted"]:
        if show["fitted_band"]:
            _band(fig, t, s.get("band_posterior"), POSTERIOR_COLOUR, "Fitted 95% band", "fit")
        fig.add_trace(go.Scatter(x=t, y=s["conc_posterior"], mode="lines",
                                 line=dict(color=POSTERIOR_COLOUR, width=2.5),
                                 name=f"Fitted ({entry['model']})", legendgroup="fit"))
    by_model = {r["model"]: r for r in (overlay or {}).get("results", [])}
    dotted = [m for m in overlay_models if m != entry["model"]]
    if other is not None:
        by_model[other["model"]] = other
        if other["model"] not in dotted:
            dotted.append(other["model"])
    for m in dotted:
        r = by_model.get(m)
        if not r or not r.get("ok") or not r.get("series"):
            continue
        fig.add_trace(go.Scatter(x=x(r["series"]["time_h"]),
                                 y=r["series"]["conc_posterior"], mode="lines",
                                 line=dict(color=overlay_colour(order, m), width=2,
                                           dash="dot"),
                                 name=f"Fitted ({m})", legendgroup=f"overlay_{m}"))
    if f and show["forecast"]:
        # from the first forecast dose; the fitted curve runs up to it
        keep = [i for i, h in enumerate(f["time_h"]) if h >= f["next_dose_h"] - 1e-9]
        ft = x([f["time_h"][i] for i in keep])
        if show["forecast_band"] and f.get("low"):
            _band(fig, ft, {"low": [f["low"][i] for i in keep],
                            "high": [f["high"][i] for i in keep]},
                  PROJECTION_COLOUR, "Forecast 95% band", "forecast")
        fig.add_trace(go.Scatter(x=ft, y=[f["conc"][i] for i in keep], mode="lines",
                                 line=dict(color=PROJECTION_COLOUR, width=2.5, dash="dash"),
                                 name=("Forecasted concentration"
                                       + (f" ({other['model']})" if other else "")),
                                 legendgroup="forecast"))
    if show["levels"] and response.get("levels"):
        fig.add_trace(go.Scatter(x=x([l["time_h"] for l in response["levels"]]),
                                 y=[l["conc_mg_l"] for l in response["levels"]],
                                 mode="markers", marker=dict(color=LEVEL_COLOUR, size=10),
                                 name="Measured levels"))

    fig = _style(fig, "Time" if response["regimen_given"].get("time_origin")
                 else "Hours after the first dose", "Vancomycin (mg/L)")

    # Leave headroom above the highest curve and the peak line for the labels.
    ys = [v for t in fig.data if t.y is not None for v in t.y if v is not None]
    top = max(ys + [peak_line or 0.0, 1.0])
    # ...and a thin strip below zero for the regimen labels, so they never sit on
    # a curve (every curve starts from zero at the first dose). The strip is
    # narrower than one tick step, so no negative tick label appears.
    fig.update_yaxes(range=[-top * 0.08, top * 1.15])
    # Each regimen named in its curve's colour where it begins, along the bottom
    # of the plot (clear of the legend above it): the current one at the first
    # dose, the forecast where it starts.
    labels = []
    if show["fitted"]:
        labels.append((s["time_h"][0], given_label(response), POSTERIOR_COLOUR))
    if f and show["forecast"]:
        labels.append((f["next_dose_h"], forecast_label(f), PROJECTION_COLOUR))
    for hours, text, colour in labels:
        fig.add_annotation(x=x([hours])[0], y=0.0, yref="paper", yanchor="bottom",
                           xanchor="left", xshift=6, yshift=4,
                           text=f"<b>{text}</b>", showarrow=False,
                           font=dict(size=14, color=colour))
    fig.update_layout(height=480)
    return fig


def auc_grid_html(rows: Sequence[Tuple[str, Mapping]], target: Mapping) -> str:
    """Steady-state AUC24 readouts as a two-column grid: a bold "<label> AUC24" on
    the left; on the right the value, coloured against the target window, with
    its 90% interval underneath."""
    cells = []
    for label, exposure in rows:
        auc = exposure.get("auc24") if exposure else None
        if auc is None:
            continue
        colour = (BELOW_COLOUR if auc < target["low"] else
                  ABOVE_COLOUR if auc > target["high"] else TARGET_COLOUR)
        ci = exposure.get("auc24_ci")
        ci_text = (f"<div style='font-size:0.8em; opacity:0.75;'>"
                   f"90% CI {ci[0]:.0f}-{ci[1]:.0f}</div>"
                   if ci and ci[0] is not None else "")
        cells.append(
            f"<div style='font-weight:700;'>{label} AUC<sub>24</sub></div>"
            f"<div style='text-align:right;'><span style='color:{colour}; "
            f"font-weight:700; font-size:1.35em;'>{auc:.0f}</span> "
            f"<span style='font-size:0.8em;'>mg·h/L</span>{ci_text}</div>")
    return ("<div style='display:grid; grid-template-columns:auto 1fr; "
            "column-gap:0.6em; row-gap:0.7em; align-items:start;'>"
            + "".join(cells) + "</div>")


# ======================================================================
# One interval: Sawchuk-Zaske one-compartment vs the Bayesian model
# ======================================================================
def compartment_figure(response: Mapping, model: Optional[str] = None
                       ) -> Optional[go.Figure]:
    """The two-level one-compartment curve against the fitted model over one
    interval, with the two levels the calculation used placed at their time into
    that interval. Returns None when the two-level method is unavailable."""
    sz = response["sawchuk_zaske"]
    if not sz.get("available") or not sz.get("curve"):
        return None
    entry = result_for(response, model)
    ss = entry.get("steady_state_interval")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sz["curve"]["time_into_interval_h"], y=sz["curve"]["conc"],
                             mode="lines", fill="tozeroy",
                             fillcolor=_rgba(ONE_CMT_COLOUR, 0.18),
                             line=dict(color=ONE_CMT_COLOUR, width=3, dash="dash"),
                             name="One-compartment (two-level)"))
    if ss:
        fig.add_trace(go.Scatter(x=ss["time_into_interval_h"], y=ss["conc"], mode="lines",
                                 line=dict(color=LEVEL_COLOUR, width=3),
                                 name=f"{entry['model']}, day-{ss['day']} interval"))

    used = sz.get("levels_used") or []
    if used:
        fig.add_trace(go.Scatter(x=[l["hours_into_interval"] for l in used],
                                 y=[l["conc_mg_l"] for l in used], mode="markers",
                                 marker=dict(color="#ecf0f1", size=12, symbol="x",
                                             line=dict(width=2, color="rgba(255,255,255,0.8)")),
                                 name="Levels used"))
    # No figure title: with the legend above the plot the two would compete for
    # the same space. The page heads the chart instead.
    fig = _style(fig, "Hours since the start of the dose", "Vancomycin (mg/L)")
    fig.update_layout(height=340)
    return fig


def two_level_table(response: Mapping, model: Optional[str] = None) -> Optional[pd.DataFrame]:
    """The two-level results beside the Bayesian fit's: AUC24, clearance,
    half-life and volume, with the Bayesian difference as a percentage of the
    two-level value (the engine's own rule). Half-life and volume have no single
    Bayesian counterpart in a two-compartment model, so those cells are blank.
    Returns None when the two-level method is unavailable."""
    sz = response["sawchuk_zaske"]
    if not sz.get("available"):
        return None
    entry = result_for(response, model)
    weight = (response.get("patient") or {}).get("weight_kg")

    def diff(bayes, ref):
        return f"{100.0 * (bayes - ref) / ref:+.1f}%"

    auc_b, cl_b = entry["current"]["auc24"], entry["parameters"]["CL"]
    vol = f"{sz['volume_l']:.1f} L"
    if weight:
        vol += f" ({sz['volume_l'] / weight:.2f} L/kg)"
    return pd.DataFrame([
        {"": "AUC24 (mg·h/L)", "Two-level": f"{sz['auc24']:.0f}",
         f"Bayesian ({entry['model']})": f"{auc_b:.0f}", "Difference": diff(auc_b, sz["auc24"])},
        {"": "Clearance (L/h)", "Two-level": f"{sz['clearance_l_h']:.2f}",
         f"Bayesian ({entry['model']})": f"{cl_b:.2f}",
         "Difference": diff(cl_b, sz["clearance_l_h"])},
        {"": "Half-life (h)", "Two-level": f"{sz['half_life_h']:.2f}",
         f"Bayesian ({entry['model']})": "-", "Difference": "-"},
        {"": "Volume of distribution", "Two-level": vol,
         f"Bayesian ({entry['model']})": "-", "Difference": "-"},
    ])


# ======================================================================
# Model comparison
# ======================================================================
def model_comparison_figure(response: Mapping,
                            x_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    """Estimated AUC24 per model with its 90% interval, against the target window.

    ``x_range`` fixes the axis, so the chart looks the same from case to case;
    intervals reaching past it are cut off at the edge.
    """
    rows = []
    for e in response["results"]:
        if not e.get("ok"):
            continue
        ci = e["current"].get("auc24_ci") or [None, None]
        rows.append((e["model"] + (" *" if e["out_of_population"] else ""),
                     e["current"]["auc24"], ci[0], ci[1]))
    if not rows:
        raise ValueError("No model produced a result for this case.")
    target = response["target"]
    names = [r[0] for r in rows][::-1]
    fig = go.Figure()
    fig.add_vrect(x0=target["low"], x1=target["high"], fillcolor=TARGET_COLOUR,
                  opacity=0.15, layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=[r[1] for r in rows][::-1], y=names, mode="markers",
        marker=dict(size=11, color=MEAN_COLOUR),
        error_x=dict(type="data", symmetric=False,
                     array=[(r[3] - r[1]) if r[3] is not None else 0 for r in rows][::-1],
                     arrayminus=[(r[1] - r[2]) if r[2] is not None else 0 for r in rows][::-1],
                     color=MEAN_COLOUR, thickness=2),
        name="Bayesian AUC24",
        hovertemplate="%{y}: %{x:.0f} mg·h/L<extra></extra>"))
    sz = response.get("sawchuk_zaske") or {}
    if sz.get("available"):
        fig.add_vline(x=sz["auc24"], line_dash="dash", line_color=LEVEL_COLOUR, line_width=2,
                      annotation_text=f"two-level {sz['auc24']:.0f}",
                      annotation_font=dict(color=LEVEL_COLOUR))
    fig = _style(fig, "Estimated AUC24 on the current regimen (mg·h/L)", "", legend=False)
    fig.update_layout(hovermode="closest", height=60 + 45 * len(rows))
    fig.update_yaxes(rangemode="normal")
    if x_range:
        fig.update_xaxes(range=list(x_range), rangemode="normal")
    return fig


def comparison_table(response: Mapping) -> pd.DataFrame:
    """One row per model: clearance, AUC24 on the current regimen, and the
    recommended total daily dose (the interval is left to the clinician)."""
    rows = []
    for r in response["results"]:
        ok = r.get("ok")
        rows.append({
            "Model": r["model"],
            "CL (L/h)": f"{r['parameters']['CL']:.2f}" if ok else "-",
            "AUC24 now": f"{r['current']['auc24']:.0f}" if ok else "-",
            "Recommended": (f"{r['recommendation']['total_daily_dose_mg']:.0f} mg/day"
                            if ok else "no fit"),
        })
    return pd.DataFrame(rows)


# ======================================================================
# Tables
# ======================================================================
PARAMETER_LABELS = {
    "CL": "Clearance (L/h)",
    "V": "Volume (L)",
    "V1": "Central volume (L)",
    "Q": "Intercompartmental clearance (L/h)",
    "V2": "Peripheral volume (L)",
}


def parameter_table(entry: Mapping) -> pd.DataFrame:
    """Population (prior) against individual (fitted) parameters."""
    pop, ind = entry["population_parameters"], entry["parameters"]
    return pd.DataFrame([{"Parameter": PARAMETER_LABELS.get(k, k),
                          "Population": pop[k], "Individual": ind[k]} for k in ind])


def uncertainty_table(entry: Mapping) -> pd.DataFrame:
    """Prior against posterior CV for each parameter with between-subject variability."""
    return pd.DataFrame([{"Parameter": PARAMETER_LABELS.get(r["parameter"], r["parameter"]),
                          "Prior CV (%)": r["prior_cv_pct"],
                          "Post-fit CV (%)": r["posterior_cv_pct"]}
                         for r in entry.get("parameter_uncertainty") or []])


def regimen_options(entry: Mapping) -> List[Dict[str, Any]]:
    """Every regimen the engine optimised: one per interval, shortest first, then
    the continuous infusion when it was requested. Each carries a ``label``."""
    intermittent = sorted([entry["recommendation"]] + list(entry.get("alternatives") or []),
                          key=lambda e: e["interval_h"])
    out = [dict(e, label=f"q{e['interval_h']:g}hr") for e in intermittent]
    if entry.get("continuous_infusion"):
        out.append(dict(entry["continuous_infusion"], label="Continuous infusion"))
    return out


def _percent(p: Optional[float]) -> str:
    return "-" if p is None else f"{100.0 * p:.1f}%"


def regimen_table(options: Sequence[Mapping], target: Mapping) -> pd.DataFrame:
    """The dosing table: optimal dose per regimen and its probabilities, as text."""
    peak = target.get("peak_threshold_mg_l", 50.0)
    rows = []
    for e in options:
        d = e.get("distribution") or {}
        dose = (f"{e['total_daily_dose_mg']:.0f} mg/24 hr" if e.get("continuous")
                else f"{e['dose_mg']:.0f} mg")
        rows.append({
            "Regimen": e["label"],
            "Optimal dose": dose,
            "%PTA": _percent(d.get("prob_in_target", e.get("prob_in_target"))),
            f"AUC < {target['low']:g} Risk": _percent(
                d.get("prob_below_target", e.get("prob_below_target"))),
            f"AUC > {target['high']:g} Risk": _percent(
                d.get("prob_above_target", e.get("prob_above_target"))),
            f"Peak > {peak:g} mg/L Risk": _percent(d.get("prob_peak_above")),
        })
    return pd.DataFrame(rows)
