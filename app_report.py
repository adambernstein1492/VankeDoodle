"""A one-page PDF summary of a VankeDoodle evaluation.

``build_report`` takes the JSON response from ``vankedoodle.evaluate`` (plus the
entry and exposure that the page forecasts) and returns the bytes of a single
US-Letter page. Nothing clinical is computed here: every number is read from the
response. The page is drawn with matplotlib alone, so the figure is vector and
there is no browser or image-export dependency.

Layout, top to bottom: title and disclaimer; patient and PK estimate; the doses
given and the levels measured against the model; the concentration-time figure;
current exposure; the model-suggested regimen and the alternatives; notes; and a
footer with the targets used and the model's source.
"""
from __future__ import annotations

import io
import textwrap
from datetime import datetime
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates                       # noqa: E402
from matplotlib.backends.backend_pdf import FigureCanvasPdf   # noqa: E402
from matplotlib.figure import Figure                    # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle   # noqa: E402
import pandas as pd                                     # noqa: E402

PAGE_W, PAGE_H = 8.5, 11.0          # inches (US Letter)
MARGIN = 0.5
INK = "#1f2933"
MUTED = "#5f6b7a"
RULE = "#c9d1d9"
ACCENT = "#0b6e99"
WARN_BG, WARN_EDGE = "#fdecea", "#c0392b"
GOOD, LOW, HIGH = "#1e8449", "#b7950b", "#c0392b"
PRIOR_C, FIT_C, FORECAST_C, LEVEL_C = "#95a5a6", "#1e8449", "#0b6e99", "#c0392b"

NOTE_CLEARANCE = 0.12          # inches kept clear between the notes and the footer
MAX_LEVEL_ROWS = 5             # most recent levels listed; all are plotted

DISCLAIMER = (
    "NOT FOR CLINICAL DECISION-MAKING. This report comes from research software that "
    "has not been validated for clinical use. All values are model-based estimates "
    "that may be wrong. Any dosing decision must be made by a qualified clinician "
    "following institutional protocols, with independent verification of every dose."
)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _clock(response: Mapping, hours: float) -> datetime:
    stamp = (pd.Timestamp(response["regimen_given"]["time_origin"])
             + pd.Timedelta(hours=float(hours)))
    return stamp.round("s").to_pydatetime()


def _fmt_dt(d: datetime) -> str:
    return d.strftime("%m/%d/%Y %H:%M")


def _pct(p: Optional[float]) -> str:
    return "-" if p is None else f"{100 * p:.0f}%"


def _auc_colour(auc: float, target: Mapping) -> str:
    return LOW if auc < target["low"] else HIGH if auc > target["high"] else GOOD


def _age_text(patient: Mapping) -> str:
    days = patient.get("postnatal_age_days")
    years = patient.get("age_years")
    if days is not None and days < 365.25:
        return f"{days:.0f} days"
    return f"{years:.1f} years" if years is not None else "-"


def _regimen_text(e: Mapping, short: bool = False) -> str:
    if e.get("continuous"):
        return f"{e['total_daily_dose_mg']:.0f} mg/24 h continuous infusion"
    if short:
        return f"{e['dose_mg']:.0f} mg q{e['interval_h']:g}h"
    return (f"{e['dose_mg']:.0f} mg IV every {e['interval_h']:g} h, "
            f"infused over {e['infusion_h']:g} h")


class _Page:
    """A Letter page with a cursor measured in inches from the top."""

    def __init__(self) -> None:
        self.fig = Figure(figsize=(PAGE_W, PAGE_H))
        FigureCanvasPdf(self.fig)
        self.y = MARGIN

    # figure coordinates from inches (origin top-left)
    @staticmethod
    def fx(x: float) -> float:
        return x / PAGE_W

    @staticmethod
    def fy(y: float) -> float:
        return 1.0 - y / PAGE_H

    def text(self, x: float, y: float, s: str, size: float = 8.5, weight: str = "normal",
             color: str = INK, ha: str = "left", va: str = "top", **kw) -> None:
        self.fig.text(self.fx(x), self.fy(y), s, fontsize=size, fontweight=weight,
                      color=color, ha=ha, va=va, **kw)

    def wrapped(self, x: float, y: float, s: str, width_in: float, size: float = 8.0,
                color: str = INK, weight: str = "normal", line_h: Optional[float] = None
                ) -> float:
        """Text wrapped to ``width_in``; returns the height used (inches)."""
        chars = max(20, int(width_in * 72 / (size * 0.5)))
        lines = textwrap.wrap(s, chars) or [""]
        lh = line_h or size * 1.35 / 72
        for i, line in enumerate(lines):
            self.text(x, y + i * lh, line, size=size, color=color, weight=weight)
        return len(lines) * lh

    def rule(self, y: float, x0: float = MARGIN, x1: float = PAGE_W - MARGIN,
             color: str = RULE, lw: float = 0.8) -> None:
        self.fig.add_artist(matplotlib.lines.Line2D(
            [self.fx(x0), self.fx(x1)], [self.fy(y), self.fy(y)], color=color, lw=lw))

    def box(self, x: float, y: float, w: float, h: float, face: str, edge: str,
            lw: float = 1.0) -> None:
        self.fig.add_artist(FancyBboxPatch(
            (self.fx(x), self.fy(y + h)), w / PAGE_W, h / PAGE_H,
            boxstyle="round,pad=0,rounding_size=0.004", transform=self.fig.transFigure,
            facecolor=face, edgecolor=edge, linewidth=lw))

    def heading(self, x: float, y: float, s: str, width: float) -> float:
        self.text(x, y, s.upper(), size=8.5, weight="bold", color=ACCENT)
        self.rule(y + 0.17, x, x + width, color=ACCENT, lw=0.6)
        return 0.25

    def table(self, x: float, y: float, cols: Sequence[Tuple[str, float, str]],
              rows: Sequence[Sequence[str]], size: float = 7.6, row_h: float = 0.165,
              bold_rows: Sequence[int] = (), row_colours: Optional[Sequence] = None
              ) -> float:
        """A plain table. ``cols`` are (title, width_in, align). Returns height used."""
        cx = x
        for title, w, align in cols:
            ax = cx + (w if align == "right" else w / 2 if align == "center" else 0)
            self.text(ax, y, title, size=size - 0.4, weight="bold", color=MUTED, ha=align)
            cx += w
        total_w = sum(c[1] for c in cols)
        self.rule(y + row_h - 0.02, x, x + total_w, lw=0.6)
        for r, row in enumerate(rows):
            ry = y + (r + 1) * row_h + 0.01
            if r in bold_rows:
                self.fig.add_artist(Rectangle(
                    (self.fx(x - 0.04), self.fy(ry + row_h - 0.035)),
                    (total_w + 0.08) / PAGE_W, row_h / PAGE_H, transform=self.fig.transFigure,
                    facecolor="#e8f4f8", edgecolor="none", zorder=0))
            cx = x
            for c, ((_, w, align), cell) in enumerate(zip(cols, row)):
                ax = cx + (w if align == "right" else w / 2 if align == "center" else 0)
                colour = (row_colours[r][c] if row_colours and row_colours[r]
                          and row_colours[r][c] else INK)
                self.text(ax, ry, cell, size=size, ha=align, color=colour,
                          weight="bold" if r in bold_rows else "normal")
                cx += w
        return (len(rows) + 1) * row_h + 0.05

    def to_pdf(self) -> bytes:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="pdf")
        return buf.getvalue()


# ----------------------------------------------------------------------------
# the concentration-time figure
# ----------------------------------------------------------------------------
def _draw_figure(page: _Page, top: float, height: float, response: Mapping,
                 entry: Mapping, forecast_entry: Mapping,
                 trough_band: Optional[Tuple[float, float]],
                 peak_line: Optional[float]) -> None:
    # room above the axes for the legend and below them for the time labels
    left, right = MARGIN + 0.45, PAGE_W - MARGIN - 0.05
    ax_top, ax_bottom = top + 0.28, top + height - 0.40
    ax = page.fig.add_axes([page.fx(left), page.fy(ax_bottom),
                            (right - left) / PAGE_W, (ax_bottom - ax_top) / PAGE_H])
    s, f = entry["series"], forecast_entry.get("forecast")
    t = [_clock(response, h) for h in s["time_h"]]
    ymax = max(s["conc_posterior"] + s["conc_prior"])
    if trough_band:
        ax.axhspan(*trough_band, color="#999999", alpha=0.12, lw=0, zorder=0)
    if s.get("band_posterior"):
        ax.fill_between(t, s["band_posterior"]["low"], s["band_posterior"]["high"],
                        color=FIT_C, alpha=0.15, lw=0)
    ax.plot(t, s["conc_prior"], color=PRIOR_C, lw=1.2, label="Population prior")
    ax.plot(t, s["conc_posterior"], color=FIT_C, lw=1.8, label=f"Fitted ({entry['model']})")
    if f:
        keep = [i for i, h in enumerate(f["time_h"]) if h >= f["next_dose_h"] - 1e-9]
        ft = [_clock(response, f["time_h"][i]) for i in keep]
        if f.get("low"):
            ax.fill_between(ft, [f["low"][i] for i in keep], [f["high"][i] for i in keep],
                            color=FORECAST_C, alpha=0.13, lw=0)
            ymax = max(ymax, max(f["high"][i] for i in keep))
        name = "Forecast" + (f" ({forecast_entry['model']})"
                             if forecast_entry["model"] != entry["model"] else "")
        ax.plot(ft, [f["conc"][i] for i in keep], color=FORECAST_C, lw=1.6, ls="--",
                label=name)
        ax.axvspan(ft[0], ft[-1], color="#cccccc", alpha=0.12, lw=0, zorder=0)
        ymax = max(ymax, max(f["conc"][i] for i in keep))
    if response.get("levels"):
        lt = [_clock(response, l["time_h"]) for l in response["levels"]]
        ax.scatter(lt, [l["conc_mg_l"] for l in response["levels"]], color=LEVEL_C, s=22,
                   zorder=5, label="Measured levels")
        ymax = max(ymax, max(l["conc_mg_l"] for l in response["levels"]))
    if peak_line:
        ax.axhline(peak_line, color=LEVEL_C, lw=0.8, ls=":", zorder=1)
        ax.text(t[0], peak_line, f" peak {peak_line:g} mg/L", color=LEVEL_C, fontsize=6.5,
                va="top")
        ymax = max(ymax, peak_line)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_ylabel("Vancomycin (mg/L)", fontsize=7.5, color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.tick_params(labelsize=6.8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.6, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=5,
              frameon=False, handlelength=2.2, borderaxespad=0)


# ----------------------------------------------------------------------------
# the page
# ----------------------------------------------------------------------------
def build_report(response: Mapping, entry: Mapping, forecast_entry: Mapping,
                 forecast_exposure: Mapping, options: Sequence[Mapping],
                 case_id: Optional[str] = None,
                 trough_band: Optional[Tuple[float, float]] = (8.0, 12.0),
                 generated: Optional[datetime] = None) -> bytes:
    """A one-page PDF for one evaluation.

    ``entry`` is the population model's result (fit, current exposure, series);
    ``forecast_entry`` the dosing model's (the same entry unless another was
    chosen), with ``forecast_exposure`` the regimen it forecasts; ``options`` the
    rows of the alternatives table (``app_charts.regimen_options``).
    """
    page = _Page()
    W = PAGE_W - 2 * MARGIN
    x0 = MARGIN
    target = response["target"]
    peak = target.get("peak_threshold_mg_l")
    p = response["patient"]
    generated = generated or datetime.now()

    # ---------------------------------------------------------------- title
    y = MARGIN
    page.text(x0, y, "Vancomycin AUC-Guided Dosing Report", size=15, weight="bold")
    page.text(PAGE_W - MARGIN, y + 0.02, f"Generated {_fmt_dt(generated)}", size=7.5,
              color=MUTED, ha="right")
    page.text(PAGE_W - MARGIN, y + 0.17,
              f"VankeDoodle {response.get('engine_version', '')}", size=7.5, color=MUTED,
              ha="right")
    y += 0.32
    page.text(x0, y, f"Patient ID: {case_id}" if case_id else "Patient ID: not recorded",
              size=9, color=MUTED)
    y += 0.25

    # ---------------------------------------------------------------- disclaimer
    disc_h = 0.5
    page.box(x0, y, W, disc_h, WARN_BG, WARN_EDGE, lw=1.0)
    page.wrapped(x0 + 0.12, y + 0.08, DISCLAIMER, W - 0.24, size=7.6, color="#7b241c",
                 weight="bold")
    y += disc_h + 0.18

    # ---------------------------------------------------------------- patient | PK
    col_w = (W - 0.3) / 2
    xl, xr = x0, x0 + col_w + 0.3
    page.heading(xl, y, "Patient", col_w)
    page.heading(xr, y, "Pharmacokinetic estimate", col_w)
    y += 0.25
    born = (datetime.fromisoformat(p["birthdate"]).strftime("%m/%d/%Y")
            if p.get("birthdate") else "-")
    sex = {"M": "Male", "F": "Female"}.get(p.get("sex") or "", "Not recorded")
    left_rows = [("Birthdate", f"{born}  (age {_age_text(p)} at first dose)"),
                 ("Sex", sex),
                 ("Weight / height", f"{p['weight_kg']:g} kg  /  {p['height_cm']:g} cm"),
                 ("Serum creatinine", f"{p['scr_mg_dl']:.2f} mg/dL  ({p['scr_umol_l']:.0f} µmol/L)"),
                 ("CrCl (bedside Schwartz)",
                  f"{p['crcl_bedside_schwartz']:.0f} mL/min/1.73 m²")]
    if p.get("gestational_age_weeks") and p.get("postnatal_age_days", 1e9) < 365.25:
        left_rows.append(("Gestational / PMA",
                          f"{p['gestational_age_weeks']:g} wk  /  {p['pma_weeks']:.1f} wk"))
    pars, pop = entry["parameters"], entry["population_parameters"]
    vol = sum(v for k, v in pars.items() if k.startswith("V"))
    right_rows = [("Population model", f"{entry['model']} ({entry['n_compartments']}-compartment)"),
                  ("Clearance", f"{pars['CL']:.2f} L/h  ({pars['CL'] / p['weight_kg']:.3f} L/h/kg)"
                                f"   population {pop['CL']:.2f}"),
                  ("Total volume", f"{vol:.1f} L  ({vol / p['weight_kg']:.2f} L/kg)"),
                  ("Levels used", f"{entry['n_levels_used']}"
                                  + ("" if entry["n_levels_used"] else " (population prior only)")),
                  ("Uncertainty", entry.get("posterior_method") or "-")]
    for i, (k, v) in enumerate(left_rows):
        page.text(xl, y + i * 0.17, k, size=7.8, color=MUTED)
        page.text(xl + 1.45, y + i * 0.17, v, size=7.8)
    for i, (k, v) in enumerate(right_rows):
        page.text(xr, y + i * 0.17, k, size=7.8, color=MUTED)
        page.text(xr + 1.2, y + i * 0.17, v, size=7.8)
    y += max(len(left_rows), len(right_rows)) * 0.17 + 0.12

    # ---------------------------------------------------------------- doses | levels
    page.heading(xl, y, "Doses given", col_w)
    page.heading(xr, y, "Levels: measured vs model", col_w)
    y += 0.25
    g = response["regimen_given"]
    doses = sorted(g["doses"], key=lambda d: d["time_h"])
    first, last = _clock(response, doses[0]["time_h"]), _clock(response, doses[-1]["time_h"])
    amounts = {round(d["amount_mg"], 3) for d in doses}
    if len(amounts) == 1:
        given = (f"{doses[-1]['amount_mg']:.0f} mg every {g['observed_interval_h']:g} h, "
                 f"{len(doses)} dose{'s' if len(doses) != 1 else ''}")
    else:
        given = (f"{len(doses)} doses of {min(amounts):.0f}-{max(amounts):.0f} mg; most "
                 f"recent {doses[-1]['amount_mg']:.0f} mg, every {g['observed_interval_h']:g} h")
    dy = page.wrapped(xl, y, given, col_w, size=7.8)
    page.text(xl, y + dy, f"First {_fmt_dt(first)}   ·   last {_fmt_dt(last)}", size=7.8)
    page.text(xl, y + dy + 0.17,
              f"{g['total_daily_dose_mg']:.0f} mg/day  =  {g['mg_per_kg_per_day']:.0f} mg/kg/day",
              size=7.8)
    nxt = forecast_entry.get("forecast", {}).get("next_dose_h")
    if nxt is not None:
        page.text(xl, y + dy + 0.34, f"Next dose due {_fmt_dt(_clock(response, nxt))}",
                  size=7.8, weight="bold")
    left_used = dy + 0.55

    lv = response.get("levels") or []
    fit = entry.get("fit_to_levels") or {}
    rows = []
    for i, l in enumerate(lv[-MAX_LEVEL_ROWS:]):
        k = len(lv) - min(len(lv), MAX_LEVEL_ROWS) + i
        pred = fit.get("predicted", [None] * len(lv))[k]
        err = fit.get("percent_error", [None] * len(lv))[k]
        rows.append([_fmt_dt(_clock(response, l["time_h"])), f"{l['conc_mg_l']:.1f}",
                     "-" if pred is None else f"{pred:.1f}",
                     "-" if err is None else f"{err:+.0f}%"])
    if rows:
        right_used = page.table(xr, y, [("Drawn", 1.35, "left"), ("Measured", 0.8, "right"),
                                        ("Model", 0.75, "right"), ("Diff.", 0.6, "right")],
                                rows)
        if len(lv) > MAX_LEVEL_ROWS:
            page.text(xr, y + right_used,
                      f"Most recent {MAX_LEVEL_ROWS} of {len(lv)} levels shown (all are plotted).",
                      size=6.8, color=MUTED)
            right_used += 0.15
    else:
        page.text(xr, y, "No levels entered: estimates are the population prior.",
                  size=7.8, color=MUTED)
        right_used = 0.2
    y += max(left_used, right_used) + 0.12

    # ---------------------------------------------------------------- figure
    # The figure takes whatever height is left once everything below it, notes
    # included, has its room, so the warnings always fit on the one page.
    fy = PAGE_H - MARGIN - 0.42                     # where the footer starts
    fe = forecast_exposure
    # Levels that cannot match the doses: the fit is shown for what it implies,
    # but no regimen is suggested or forecast from it.
    withheld = bool(forecast_entry.get("implausible_fit"))
    notes = list(entry.get("warnings") or []) + list(fe.get("flags") or [])
    if withheld:
        notes = [n for n in notes if "cannot be reconciled" not in n]
    notes = [n for i, n in enumerate(notes) if n not in notes[:i]]
    note_chars = int((W - 0.12) * 72 / (7.2 * 0.5))
    note_lines = sum(len(textwrap.wrap("•  " + n, note_chars)) for n in notes)
    reason = next((w for w in (forecast_entry.get("warnings") or [])
                   if "cannot be reconciled" in w), "")
    reason_lines = len(textwrap.wrap(reason, int(W * 72 / (7.8 * 0.5)))) if withheld else 0
    below = (0.55                                   # exposure
             + ((0.25 + reason_lines * 7.8 * 1.35 / 72 + 0.3) if withheld else
                0.83 + (len(options) + 1) * 0.165 + 0.05 + 0.22)    # regimen + table
             + ((0.25 + note_lines * 7.2 * 1.35 / 72) if notes else 0.0)
             + NOTE_CLEARANCE + 0.02)
    fig_h = min(2.75, max(1.5, fy - (y + 0.27) - below))

    page.heading(x0, y, "Concentration over time", W)
    _draw_figure(page, y + 0.22, fig_h, response, entry,
                 {**forecast_entry, "forecast": None} if withheld else forecast_entry,
                 trough_band, peak)
    y += 0.22 + fig_h + 0.05

    # ---------------------------------------------------------------- exposure
    page.heading(x0, y, "Exposure (steady-state AUC24, target "
                 f"{target['low']:g}-{target['high']:g} mg·h/L)", W)
    y += 0.25
    cur = entry["current"]
    ci = cur.get("auc24_ci")
    page.text(xl, y, "Current regimen", size=7.8, color=MUTED)
    page.text(xl + 1.2, y, f"{cur['auc24']:.0f}", size=11, weight="bold",
              color=_auc_colour(cur["auc24"], target))
    page.text(xl + 1.6, y + 0.03,
              f"mg·h/L   90% CI {ci[0]:.0f}-{ci[1]:.0f}   P(in target) {_pct(cur.get('prob_in_target'))}"
              if ci else "mg·h/L", size=7.8)
    sz = response.get("sawchuk_zaske") or {}
    if sz.get("available"):
        diff = response.get("comparison", {}).get("percent_difference", {}).get(entry["model"])
        page.text(xr + 0.6, y + 0.03,
                  f"Two-level (Sawchuk-Zaske) check: {sz['auc24']:.0f} mg·h/L"
                  + (f"  (Bayesian {diff:+.0f}%)" if diff is not None else ""),
                  size=7.8, color=MUTED)
    y += 0.3

    # ---------------------------------------------------------------- recommendation
    if withheld:
        page.heading(x0, y, "No dose recommended", W)
        y += 0.25
        y += page.wrapped(xl, y, reason, W - 0.12, size=7.8, color=LEVEL_C)
        page.text(xl, y + 0.03, "The fitted curve and current AUC24 above show what the "
                  "levels imply, not this patient. Correct the doses, units or sample "
                  "times and generate the report again.", size=7.2, color=MUTED)
        y += 0.3
    else:
        y = _recommendation(page, y, x0, xl, W, entry, forecast_entry, fe, options,
                            target, peak)

    # ---------------------------------------------------------------- notes
    if notes:
        page.heading(x0, y, "Notes", W)
        y += 0.25
        for i, n in enumerate(notes):
            need = len(textwrap.wrap("•  " + n, note_chars)) * 7.2 * 1.35 / 72
            if y + need > fy - NOTE_CLEARANCE:         # keep to one page
                page.text(x0 + 0.12, y, f"({len(notes) - i} more note(s) not shown)",
                          size=7.0, color=MUTED)
                break
            y += page.wrapped(x0 + 0.12, y, "•  " + n, W - 0.12, size=7.2, color=INK)

    # ---------------------------------------------------------------- footer
    page.rule(fy - 0.06)
    page.wrapped(x0, fy, f"Targets: AUC24 {target['low']:g}-{target['high']:g} mg·h/L "
                 f"(aim {target['auc24']:g}); peak ceiling {peak:g} mg/L. "
                 f"Model source: {entry.get('reference', '')}", W, size=6.4, color=MUTED)
    page.text(PAGE_W - MARGIN, PAGE_H - MARGIN + 0.02, "Page 1 of 1", size=6.4,
              color=MUTED, ha="right", va="bottom")
    return page.to_pdf()


def _recommendation(page, y, x0, xl, W, entry, forecast_entry, fe, options, target,
                    peak) -> float:
    """The suggested (or test) regimen and the table of alternatives; returns the
    new y."""
    is_test = fe is not forecast_entry.get("recommendation")
    title = "Regimen evaluated" if is_test else "Model-suggested regimen"
    page.heading(x0, y, title + " (for clinician review)", W)
    y += 0.25
    fci = fe.get("auc24_ci")
    page.text(xl, y, _regimen_text(fe), size=10, weight="bold", color=ACCENT)
    extra = (f"  ·  {fe['mg_per_kg_per_day']:.0f} mg/kg/day" if fe.get("mg_per_kg_per_day")
             else "")
    page.text(xl, y + 0.2,
              f"Predicted steady-state AUC24 {fe['auc24']:.0f} mg·h/L"
              + (f" (90% CI {fci[0]:.0f}-{fci[1]:.0f})" if fci else "")
              + f"  ·  P(in target) {_pct(fe.get('prob_in_target'))}"
              + f"  ·  day-{fe['evaluation_day']} peak {fe['cmax']:.0f} mg/L"
              + ("" if fe.get("continuous") else f", trough {fe['ctrough']:.1f} mg/L")
              + extra, size=7.8)
    basis = (f"Based on {forecast_entry['model']}"
             + (" (dosing model chosen separately from the population model)."
                if forecast_entry["model"] != entry["model"] else ".")
             + (" Entered as a test regimen rather than the model's own suggestion."
                if is_test else ""))
    page.text(xl, y + 0.37, basis, size=7.2, color=MUTED)
    y += 0.58

    cols = [("Regimen", 1.25, "left"), ("Dose", 1.0, "right"), ("mg/kg/day", 0.85, "right"),
            ("AUC24", 0.7, "right"), ("%PTA", 0.7, "right"),
            (f"AUC<{target['low']:g}", 0.8, "right"), (f"AUC>{target['high']:g}", 0.8, "right"),
            (f"Peak>{peak:g}" if peak else "Peak", 0.85, "right")]
    rows, colours, bold = [], [], []
    for i, e in enumerate(options):
        d = e.get("distribution") or {}
        label = e.get("label") or _regimen_text(e, short=True)
        dose = (f"{e['total_daily_dose_mg']:.0f} mg/24 h" if e.get("continuous")
                else f"{e['dose_mg']:.0f} mg")
        rows.append([label, dose, f"{e['mg_per_kg_per_day']:.0f}", f"{e['auc24']:.0f}",
                     _pct(d.get("prob_in_target", e.get("prob_in_target"))),
                     _pct(d.get("prob_below_target", e.get("prob_below_target"))),
                     _pct(d.get("prob_above_target", e.get("prob_above_target"))),
                     _pct(d.get("prob_peak_above"))])
        colours.append([None, None, None, _auc_colour(e["auc24"], target)] + [None] * 4)
        if (abs(e["dose_mg"] - fe["dose_mg"]) < 1e-6 and abs(e["interval_h"] - fe["interval_h"]) < 1e-6
                and bool(e.get("continuous")) == bool(fe.get("continuous"))):
            bold.append(i)
    used = page.table(x0 + 0.05, y, cols, rows, bold_rows=bold, row_colours=colours)
    page.text(x0 + 0.05, y + used, "Alternatives: the dose putting each regimen's AUC24 "
              f"closest to {target['auc24']:g}. Highlighted: the regimen above, when it "
              "is one of them.", size=6.8, color=MUTED)
    return y + used + 0.22
