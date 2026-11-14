"""The one-page PDF report."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import vankedoodle as vd

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")          # app_charts builds the alternatives rows

ROOT = Path(__file__).resolve().parents[1]


def _case(**changes):
    req = vd.read_case_text((ROOT / "example_case.txt").read_text(encoding="utf8"))
    for key, value in changes.items():
        req[key] = value
    req["options"] = {**req.get("options", {}), "uncertainty": True,
                      "continuous_infusion": True, "peak_threshold_mg_l": 50.0,
                      "forecast_h": 48.0, **changes.get("options", {})}
    return req


def _report(req):
    import app_charts
    import app_report
    resp = vd.evaluate(req)
    e = resp["results"][0]
    shown = e["candidate"] if e.get("candidate") else e["recommendation"]
    return app_report.build_report(resp, e, e, shown, app_charts.regimen_options(e),
                                   case_id=req.get("case_id"))


def _pages(pdf: bytes) -> int:
    return max(int(n) for n in re.findall(rb"/Count (\d+)", pdf))


def test_report_is_a_single_pdf_page():
    pdf = _report(_case())
    assert pdf.startswith(b"%PDF") and _pages(pdf) == 1


@pytest.mark.parametrize("changes", [
    {"levels": []},
    {"options": {"candidate_regimen": {"dose_mg": 600, "interval_h": 12, "infusion_h": 1.5}}},
    {"options": {"candidate_regimen": {"dose_mg": 1300, "interval_h": 24, "infusion_h": 24}}},
])
def test_report_handles_other_cases_on_one_page(changes):
    assert _pages(_report(_case(**changes))) == 1


def test_report_stays_on_one_page_with_many_levels():
    req = _case()
    req["doses"] += [{"time": f"2026-03-04 {h:02d}:00", "amount_mg": 400, "infusion_h": 1}
                     for h in (0, 8, 16)]
    req["levels"] += [{"time": t, "conc_mg_l": c} for t, c in
                      (("2026-03-03 23:30", 7.0), ("2026-03-04 02:00", 29.0),
                       ("2026-03-04 07:30", 7.5), ("2026-03-04 10:00", 27.0),
                       ("2026-03-04 15:30", 8.0))]
    assert _pages(_report(req)) == 1


def test_disclaimer_says_not_for_clinical_use():
    import app_report
    assert app_report.DISCLAIMER.startswith("NOT FOR CLINICAL DECISION-MAKING")
