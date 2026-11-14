"""VankeDoodle -- Bayesian (model-informed) vancomycin dosing for children.

Quick start
-----------
>>> from vankedoodle import start_dose, assess
>>> start_dose(weight_kg=20, age_years=6, height_cm=115, scr=0.4).summary()
>>> a = assess(
...     patient={"weight_kg": 20, "age_years": 6, "height_cm": 115, "scr": 0.4},
...     regimen={"dose_mg": 300, "interval_h": 6, "n_doses": 4,
...              "start": "2026-01-01 08:00"},
...     levels=[("2026-01-01 23:00", 9.4)],
... )
>>> print(a.summary())

The estimation engine is maximum a posteriori Bayesian estimation against a
published pediatric population pharmacokinetic prior. Several priors ship with
the package (``vankedoodle.MODELS``); pass ``model="..."`` to choose one, and
``compare_models()`` to fit the same patient under all of them.
Nothing in this package makes clinical decisions; it produces exposure estimates
that a qualified clinician interprets.
"""
from ._version import __version__
from .core import (Dose, DosingHistory, Level, Patient, age_from_birthdate, bedside_schwartz,
                   body_surface_area, schwartz_1976, schwartz_k, to_hours)
from .models import (COLIN2019, DEFAULT_MODEL, FRYMOYER2014, LAMARRE2000, LE2013,
                     MODELS, PopPKModel, SMIT2021, XU2026,
                     cv_to_omega2, describe_models, get_model, omega_from_cv,
                     register_model)
from .pk import (CompartmentalModel, OneCompartmentModel, TwoCompartmentModel,
                 model_from_params)
from .bayes import MAPFit, fit, simulate_levels
from .dosing import (AUC_TARGET, AUC_TARGET_HIGH, AUC_TARGET_LOW, Exposure,
                     Recommendation, auc_over_window, continuous_infusion,
                     exposure_timeline,
                     initial_dose, infusion_duration, predict_exposure,
                     recommend, round_dose)
from .reference import (SawchukZaskeResult, mpe_mape, sawchuk_zaske,
                        two_level_from_history)
from .workflow import Assessment, assess, compare_models, start_dose
from .casefile import CaseFileError, read_case_text, write_case_text
from .api import (SCHEMA_VERSION, append_case, case_record, evaluate,
                  load_cases)



__all__ = [
    "CaseFileError", "read_case_text", "write_case_text",
    "Patient", "age_from_birthdate", "Dose", "Level", "DosingHistory", "bedside_schwartz",
    "schwartz_1976", "schwartz_k", "body_surface_area", "to_hours",
    "PopPKModel", "MODELS", "DEFAULT_MODEL", "SMIT2021", "XU2026", "LE2013",
    "LAMARRE2000", "COLIN2019", "FRYMOYER2014",
    "get_model", "register_model", "describe_models", "cv_to_omega2", "omega_from_cv",
    "CompartmentalModel", "OneCompartmentModel", "TwoCompartmentModel",
    "model_from_params",
    "MAPFit", "fit", "simulate_levels",
    "Exposure", "Recommendation", "predict_exposure", "recommend", "initial_dose",
    "continuous_infusion",
    "auc_over_window", "exposure_timeline", "infusion_duration", "round_dose",
    "AUC_TARGET", "AUC_TARGET_LOW", "AUC_TARGET_HIGH",
    "sawchuk_zaske", "SawchukZaskeResult", "mpe_mape", "two_level_from_history",
    "evaluate", "case_record", "append_case", "load_cases", "SCHEMA_VERSION",
    "Assessment", "assess", "compare_models", "start_dose",
    "__version__",
]
