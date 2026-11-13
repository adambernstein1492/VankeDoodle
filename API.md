# VankeDoodle JSON API

One call in, one document out. The request and response are plain JSON, so the
same contract serves a Streamlit app, a static browser build, an HTTP service or
an EHR integration. A front end should **read values out of the response rather
than recomputing anything** — that is what keeps the clinical logic in one place
and testable.

```python
from vankedoodle import evaluate
response = evaluate(request)          # dict in, dict out
```

```bash
python -m vankedoodle evaluate case.json --record cases.jsonl
cat case.json | python -m vankedoodle evaluate -
```

`schema_version` is `"1.0"`. Additive changes (new fields) will not bump it;
anything that removes or repurposes a field will.

---

## Request

```jsonc
{
  "case_id": "example-001",            // optional, echoed back and stored on the record
  "patient": {
    "weight_kg": 23.0,                 // required
    "age_years": 7.0,
    "height_cm": 122.0,
    "sex": "M",                        // only used by lamarre2000 (Schwartz k)
    "scr_mg_dl": 0.33,                 // or "scr_umol_l"; the other is derived
    "crcl": null,                      // optional: supply bedside Schwartz directly
    "gestational_age_weeks": null,     // neonates
    "postnatal_age_days": null         // neonates; also sets age_years if absent
  },

  // Either "regimen" (a repeating schedule) or "doses" (explicit list).
  "regimen": {
    "dose_mg": 450, "interval_h": 8, "n_doses": 5, "infusion_h": 1.0,
    "start": "2026-03-02 08:00",       // datetime string, or a number of hours
    "first_dose_mg": null              // a loading dose that differs from the rest
  },
  "doses": [
    {"amount_mg": 450, "time": "2026-03-02 08:00", "infusion_h": 1.0}
  ],

  "levels": [
    {"time": "2026-03-03 07:30", "conc_mg_l": 6.2}
  ],

  "options": {
    "model": "smit2021",               // a name, a list of names, or "all"
    "target_auc24": 500,
    "target_low": 400, "target_high": 600,
    "interval_h": null,                // pin the interval instead of letting it be chosen
    "candidate_regimen": {"dose_mg": 525, "interval_h": 8, "infusion_h": 1.0},
    "series": true                     // include plot-ready curves
  }
}
```

Times may be datetimes (`"YYYY-MM-DD HH:MM"`, ISO-8601, or `/` separators) or
plain numbers meaning hours after the first dose. Mixing is not required and both
give identical results. Irregular, delayed and missed doses are handled by listing
them explicitly in `doses`.

Model requirements differ — `vankedoodle.get_model(name).required` lists them, and
a model that cannot run is reported in its own entry rather than failing the request.

## Response

```jsonc
{
  "schema_version": "1.0",
  "ok": true,
  "errors": [],                        // request-level problems, written for display
  "engine_version": "0.1.0",
  "generated_at": "2026-03-04T09:15:00+10:00",
  "case_id": "example-001",

  "patient": { /* inputs plus derived crcl_bedside_schwartz, pma_weeks,
                  both creatinine units, assumed_term_gestation */ },
  "regimen_given": { "doses": [...], "observed_interval_h": 8.0,
                     "total_daily_dose_mg": 1350.0, "mg_per_kg_per_day": 58.7, ... },
  "levels": [ {"time_h": 23.5, "conc_mg_l": 6.2} ],
  "target": {"auc24": 500, "low": 400, "high": 600},

  "results": [                          // one entry per model requested
    {
      "model": "smit2021", "label": "...", "n_compartments": 2,
      "population": "...", "reference": "...",
      "ok": true, "error": null,
      "warnings": ["crcl = 152.7 is outside the 8.6-120 range ..."],
      "out_of_population": true,        // true when any warning is an extrapolation
      "n_levels_used": 3,
      "parameters": {"CL": 3.1, "V1": 9.3, "Q": 1.5, "V2": 11.5},
      "population_parameters": { ... }, // the prior, for contrast
      "clearance_ml_per_h_per_kg": 134.8,
      "eta": [0.17, -0.02],
      "fit_to_levels": {"observed": [...], "predicted": [...],
                        "percent_error": [...], "weighted_residual": [...]},
      "current":        { /* exposure on the regimen as given */ },
      "recommendation": { /* exposure on the recommended regimen */ },
      "alternatives":  [ /* the other intervals considered, best first */ ],
      "candidate":       { /* exposure on options.candidate_regimen, or null */ },
      "series": {"time_h": [...], "conc_prior": [...], "conc_posterior": [...],
                 "projection_time_h": [...], "projection_conc": [...]}
    }
  ],

  "sawchuk_zaske": {
    "available": true, "reason": null,
    "dose_mg": 450.0, "interval_h": 8.0, "infusion_h": 1.0,
    "levels_used": [{"time_h": 26.0, "hours_into_interval": 2.0, "conc_mg_l": 30.2},
                    {"time_h": 31.5, "hours_into_interval": 7.5, "conc_mg_l": 8.3}],
    "ke": 0.2348, "half_life_h": 2.95, "volume_l": 12.4,
    "clearance_l_h": 2.91, "cmax": 33.9, "cmin": 7.2,
    "auc_tau": 154.0, "auc24": 462.0
  },

  "comparison": {
    "sawchuk_zaske_auc24": 462.0,
    "bayesian_auc24": {"smit2021": 514.0, "colin2019": 452.7},
    "percent_difference": {"smit2021": 11.3, "colin2019": -2.0}
  }
}
```

### Every exposure object

`current`, `recommendation`, each entry of `alternatives`, and `candidate` share
one shape:

| field | meaning |
|---|---|
| `dose_mg`, `interval_h`, `infusion_h` | the regimen |
| `total_daily_dose_mg`, `mg_per_kg_per_day` | derived daily dose |
| `auc24` | steady-state AUC₂₄ = total daily dose / CL |
| `auc24_ci` | 90% interval from the Laplace posterior, or `null` |
| `prob_in_target` | posterior probability of landing in the target window |
| `in_target` | whether `auc24` falls in `[target.low, target.high]` |
| `auc24_day`, `evaluation_day` | actual AUC over day 3 of therapy |
| `cmax`, `ctrough` | peak and trough on that day |
| `flags` | dose-ceiling and out-of-range notices, written for display |

### Errors

Request-level problems (no doses, an unknown model, an unknown patient field, a
level before the first dose) return `{"ok": false, "errors": [...]}` with no
`results`. Problems specific to one model (a covariate that model needs but the
case lacks) appear as `{"ok": false, "error": "..."}` inside that model's entry
while the others still run. Only genuinely malformed input — a request that is
not a JSON object — raises.

### Display rules worth honouring

1. Never show `auc24` without `auc24_ci` or `prob_in_target` beside it.
2. Show `warnings` next to the number they qualify, not in a footer.
3. Name the `model` wherever a number from it appears; the shipped models can
   disagree by 40% on the same patient.
4. When `sawchuk_zaske.available` is false, show `reason` — it tells the user
   what to draw next, and it is written for a clinician.

### Payload size

`series` adds roughly 40 KB per model. Request curves for the one model being
displayed and `"series": false` for a `"model": "all"` comparison table.

---

## Study capture

Every evaluation can be flattened to one row and appended to a JSON Lines file:

```python
from vankedoodle import append_case, case_record, evaluate
response = evaluate(request)
append_case("cases.jsonl", case_record(response, model="smit2021",
                                       extra={"study_id": "PCH-07"}))
```

or `--record cases.jsonl --record-model smit2021` from the CLI. The row carries
the covariates, the regimen, both AUC estimates with their difference, the number
of levels used, and whether the model was extrapolating — which is the input to
Aims 1 and 3 without retyping anything. `load_cases(path)` reads it back;
`vankedoodle.api.CASE_RECORD_FIELDS` lists the columns.
