"""The plain-text patient file: reading, writing, and the round trip between them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import vankedoodle as vd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TXT = ROOT / "example_case.txt"
EXAMPLE_JSON = ROOT / "example_case.json"


def _auc(request, model="smit2021"):
    req = {**request, "options": {"model": model, "series": False}}
    return vd.evaluate(req)["results"][0]["current"]["auc24"]


def test_example_file_matches_the_json_example():
    from_txt = vd.read_case_text(EXAMPLE_TXT.read_text(encoding="utf8"))
    assert from_txt["options"]["model"] == "smit2021"
    assert from_txt["patient"]["birthdate"] == "2019-03-02"
    as_json = json.loads(EXAMPLE_JSON.read_text(encoding="utf8"))
    assert _auc(from_txt) == pytest.approx(_auc(as_json), rel=1e-9)


def test_writing_and_reading_round_trip_exactly():
    text = EXAMPLE_TXT.read_text(encoding="utf8")
    again = vd.write_case_text(vd.read_case_text(text))
    assert again == text
    assert vd.read_case_text(again) == vd.read_case_text(text)


def test_a_regular_schedule_is_written_dose_by_dose():
    req = {"patient": {"birthdate": "2019-03-02", "weight_kg": 23, "height_cm": 122,
                       "scr_mg_dl": 0.33},
           "regimen": {"dose_mg": 450, "interval_h": 8, "n_doses": 3, "infusion_h": 1,
                       "start": "2026-03-02 22:00"},
           "levels": [{"time": "2026-03-03 21:30", "conc_mg_l": 6.2}],
           "options": {"model": "xu2026"}}
    text = vd.write_case_text(req)
    back = vd.read_case_text(text)
    assert [d["time"] for d in back["doses"]] == ["2026-03-02 22:00", "2026-03-03 06:00",
                                                  "2026-03-03 14:00"]
    assert "03/03/2026   06:00  450" in text
    assert back["options"]["model"] == "xu2026"


HAND_WRITTEN = """\
patient id: bed 4
BIRTHDATE: 3/2/2019
sex: male
weight: 23
height (cm): 122
creatinine (µmol/L): 29
gestational age (weeks):
model: SMIT2021

[doses]
03/02/2026, 08:00, 450, 1
3/2/2026\t16:00\t450
2026-03-03 00:00 450 1.5   # a slower infusion

[ Levels ]
03/03/2026 07:30 6.2
"""


def test_reading_is_forgiving():
    r = vd.read_case_text(HAND_WRITTEN)
    assert r["case_id"] == "bed 4"
    p = r["patient"]
    assert p == {"birthdate": "2019-03-02", "sex": "M", "weight_kg": 23.0,
                 "height_cm": 122.0, "scr_umol_l": 29.0}
    assert r["options"]["model"] == "smit2021"
    assert [d["infusion_h"] for d in r["doses"]] == [1.0, 1.0, 1.5]   # default of 1 h
    assert r["levels"] == [{"time": "2026-03-03 07:30", "conc_mg_l": 6.2}]
    assert vd.evaluate({**r, "options": {"model": "smit2021", "series": False}})["ok"]


def test_vancydoodle_files_load():
    text = """Birthdate: 2016-05-15
Height: 140.0
Weight: 35.0
Creatinine: 0.5
Model: Smit2021

[Doses]
# Format: YYYY-MM-DD HH:MM, dose_in_mg, infusion_time_in_hours
2026-09-20 08:00, 500, 1.0
2026-09-20 16:00, 500, 1.0

[Levels]
2026-09-21 07:30, 9.5
"""
    r = vd.read_case_text(text)
    assert r["patient"]["scr_mg_dl"] == 0.5 and len(r["doses"]) == 2
    assert r["options"]["model"] == "smit2021"


def test_every_problem_is_reported_with_its_line():
    bad = """Birthdate: 13/40/2019
Weight (lb): 50
Model: smith
[Doses]
03/02/2026 8:00
03/02/2026 25:00 450
[Levels]
03/03/2026 07:30 abc
[Notes]
"""
    with pytest.raises(vd.CaseFileError) as err:
        vd.read_case_text(bad)
    problems = err.value.problems
    assert problems[0].startswith("Line 1:") and "MM/DD/YYYY" in problems[0]
    assert any(p.startswith("Line 2:") and "kg" in p for p in problems)
    assert any(p.startswith("Line 3:") and "smit2021" in p for p in problems)
    assert any(p.startswith("Line 5:") and "dose" in p for p in problems)
    assert any(p.startswith("Line 6:") and "HH:MM" in p for p in problems)
    assert any(p.startswith("Line 8:") and "not a number" in p for p in problems)
    assert any(p.startswith("Line 9:") and "section" in p for p in problems)
    # a field already reported as bad is not reported again as missing
    assert "Missing 'Birthdate'" not in problems
    assert "Missing 'Height (cm)'" in problems


def test_writing_needs_clock_times_and_a_birthdate():
    with pytest.raises(ValueError, match="birthdate"):
        vd.write_case_text({"patient": {"weight_kg": 20}})
    with pytest.raises(ValueError, match="clock times"):
        vd.write_case_text({"patient": {"birthdate": "2019-01-01", "weight_kg": 20,
                                        "height_cm": 110, "scr_mg_dl": 0.3},
                            "doses": [{"time": 0.0, "amount_mg": 400}]})


def test_gestational_age_is_kept_when_given():
    req = {"patient": {"birthdate": "2026-02-20", "weight_kg": 3.4, "height_cm": 50,
                       "scr_mg_dl": 0.6, "gestational_age_weeks": 34},
           "doses": [{"time": "2026-03-02 08:00", "amount_mg": 50, "infusion_h": 1}]}
    text = vd.write_case_text(req)
    assert "Gestational age (weeks): 34" in text
    assert vd.read_case_text(text)["patient"]["gestational_age_weeks"] == 34.0
