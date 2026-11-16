"""Population pharmacokinetic models used as Bayesian priors.

A model is a :class:`PopPKModel`: a function from patient covariates to typical
parameter values, the between-subject variance-covariance of its random effects,
and a residual error model.  Structural form is inferred from the parameter
names it returns -- ``{CL, V}`` for one compartment, ``{CL, V1, Q, V2}`` for
two -- so estimation and dose selection are the same code either way.

Adding a model means appending one :class:`PopPKModel` to :data:`MODELS`.

Reported variability is converted to the variance scale in one place
(:func:`cv_to_omega2`), using CV = sqrt(exp(omega^2) - 1).  Where a paper
tabulates the variances themselves, those are used directly and the conversion
is bypassed; each model's ``notes`` records which.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import (DAYS_PER_YEAR, WEEKS_PER_YEAR, Patient, body_surface_area,
                   schwartz_1976)


def cv_to_omega2(cv: float) -> float:
    """Convert a log-normal CV (as a fraction, e.g. 0.287) to a variance omega^2."""
    return float(np.log(1.0 + cv ** 2))


def omega_from_cv(cvs: Sequence[float], correlation: Optional[np.ndarray] = None
                  ) -> np.ndarray:
    """Build a variance-covariance matrix from CVs and an optional correlation matrix."""
    var = np.array([cv_to_omega2(c) for c in cvs], dtype=float)
    sd = np.sqrt(var)
    if correlation is None:
        return np.diag(var)
    return np.asarray(correlation, dtype=float) * np.outer(sd, sd)


@dataclass(frozen=True)
class PopPKModel:
    """A population model with log-normal between-subject variability.

    ``typical`` maps a :class:`Patient` to the typical parameter values;
    ``eta_params`` names the parameters carrying between-subject variability, in
    the row/column order of ``omega``.
    """
    name: str
    label: str
    reference: str
    typical: Callable[[Patient], Dict[str, float]]
    eta_params: Sequence[str]
    omega: np.ndarray                 # variance-covariance of the etas
    sigma_prop: float                 # proportional residual error SD (fraction)
    sigma_add: float = 0.0            # additive residual error SD (mg/L)
    required: Sequence[str] = ()      # patient covariates the model needs
    covariate_ranges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    population: str = ""
    notes: str = ""
    # Optional override for models whose random effects do not sit one-to-one on
    # the returned parameters (e.g. separate etas on two components of CL that are
    # summed).  Called as individual_fn(patient, eta) with eta None or a vector in
    # the order of eta_params; must return the same keys as ``typical``.
    individual_fn: Optional[Callable[[Patient, Optional[np.ndarray]], Dict[str, float]]] = None

    # -- structure ------------------------------------------------------
    def individual(self, patient: Patient, eta: Sequence[float] | None = None
                   ) -> Dict[str, float]:
        """Typical parameters with log-normal random effects applied."""
        if self.individual_fn is not None:
            return self.individual_fn(
                patient, None if eta is None else np.asarray(eta, dtype=float))
        pars = dict(self.typical(patient))
        if eta is not None:
            for name, e in zip(self.eta_params, np.asarray(eta, dtype=float)):
                pars[name] = pars[name] * float(np.exp(e))
        return pars

    @property
    def n_eta(self) -> int:
        return len(self.eta_params)

    @property
    def n_compartments(self) -> int:
        probe = self.typical(_PROBE_PATIENT)
        return 2 if "Q" in probe else 1

    # -- covariate checking ---------------------------------------------
    def check(self, patient: Patient) -> list[str]:
        """Raise if a required covariate is missing; return range warnings."""
        missing = [c for c in self.required if patient.get(c) is None]
        if missing:
            raise ValueError(
                f"Model '{self.name}' requires {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not set on this patient. "
                f"Model population: {self.population}."
            )
        warnings = []
        for cov, (lo, hi) in self.covariate_ranges.items():
            val = patient.get(cov)
            if val is None:
                continue
            if val < lo or val > hi:
                warnings.append(
                    f"{cov} = {val:.4g} is outside the {lo:g}-{hi:g} range of the "
                    f"population behind '{self.name}' ({self.population}); "
                    "the prediction is an extrapolation."
                )
        if ("pma_weeks" in self.required and patient.gestational_age_weeks is None
                and patient.pma_weeks is not None and patient.assumed_term_gestation
                and patient.pma_weeks < PREMATURITY_RELEVANT_PMA_WEEKS):
            warnings.append(
                "Gestational age was not supplied; postmenstrual age assumes term "
                "(40 weeks). Supply gestational_age_weeks for preterm infants.")
        return warnings


# Beyond about a year of postmenstrual age, renal maturation is essentially
# complete in every model here, so assuming term gestation no longer changes the
# prediction materially and the warning would only add noise.
PREMATURITY_RELEVANT_PMA_WEEKS = 104.0


_PROBE_PATIENT = Patient(weight_kg=20.0, age_years=6.0, height_cm=115.0, scr_mg_dl=0.5,
                         postnatal_age_days=6.0 * 365.25, sex="M")


# ==========================================================================
# Smit et al. 2021 -- children and adolescents 1-18 years, obesity and renal
# dysfunction.  Two-compartment.
# ==========================================================================
SMIT_WT_REF = 22.1          # kg, median total body weight in the index population
SMIT_CRCL_REF = 100.0       # mL/min/1.73 m^2
SMIT_CRCL_CAP = 120.0       # bedside Schwartz CLcr is capped at this value


def _smit_typical(patient: Patient) -> Dict[str, float]:
    wt_ratio = patient.weight_kg / SMIT_WT_REF
    crcl = min(float(patient.crcl), SMIT_CRCL_CAP)
    return {
        "CL": 2.12 * wt_ratio ** 0.745 * (crcl / SMIT_CRCL_REF),
        "V1": 8.90 * wt_ratio ** 1.0,
        "Q": 1.55 * wt_ratio ** 0.599,
        "V2": 12.3 * wt_ratio ** 1.0,
    }


SMIT2021 = PopPKModel(
    name="smit2021",
    label="Smit 2021 (1-18 y, obesity and renal dysfunction)",
    reference=("Smit C, Goulooze SC, Bruggemann RJM, Sherwin CM, Knibbe CAJ. "
               "Dosing recommendations for vancomycin in children and adolescents "
               "with varying levels of obesity and renal dysfunction: a population "
               "pharmacokinetic study in 1892 children aged 1-18 years. "
               "AAPS J. 2021;23(3):53. doi:10.1208/s12248-021-00577-x"),
    typical=_smit_typical,
    eta_params=("CL", "V2"),
    omega=np.array([[cv_to_omega2(0.287), -0.085],
                    [-0.085, cv_to_omega2(1.10)]], dtype=float),
    # The published model's error is $SIGMA 0.0788 on ln(concentration) in the
    # NONMEM control stream (ESM1): a VARIANCE, so the SD is sqrt(0.0788) = 0.281.
    # Using 0.0789 as the SD trusted each level ~13x too much; in simulation, 90%
    # AUC intervals then covered the truth only ~50% of the time (91-92% with 0.281).
    sigma_prop=math.sqrt(0.0788),
    required=("weight_kg", "crcl"),
    covariate_ranges={"age_years": (1.0, 18.0), "weight_kg": (5.8, 188.0),
                      "crcl": (8.6, 120.0)},
    population="children 1-18 y, 5.8-188 kg, bedside Schwartz CLcr down to 8.6",
    notes=("Two-compartment. CL scales with total body weight (exponent 0.745) and "
           "linearly with bedside Schwartz CLcr capped at 120 mL/min/1.73 m2; V1 and "
           "V2 scale linearly with weight and Q with exponent 0.599. Between-subject "
           "variability was estimated on CL (28.7% CV) and V2 (110% CV) with "
           "covariance -0.085; CVs are read on the sqrt(exp(omega^2)-1) scale. "
           "Residual error 28.1% (log-domain variance 0.0788, applied here as a "
           "proportional SD). 1892 children, 5524 samples."),
)


# ==========================================================================
# Xu et al. 2026 -- children 3 months to 18 years, Australia-wide, with renal
# maturation.  Two-compartment, allometric to a 70 kg adult.
# ==========================================================================
XU_WT_REF = 70.0
XU_THETA_SCR = 0.0188       # per umol/L
XU_HILL = 3.4               # fixed from Rhodin et al.
XU_TM50_WEEKS = 47.7        # fixed from Rhodin et al.


def xu_scr_std_umol_l(age_years: float) -> float:
    """Age-adjusted reference serum creatinine (umol/L), Ceriotti-based.

    As published: SCRstd = -2.37330 - 12.91367*ln(age) + 23.93581*sqrt(age),
    with age in years.
    """
    a = float(age_years)
    return -2.37330 - 12.91367 * math.log(a) + 23.93581 * math.sqrt(a)


def _xu_typical(patient: Patient) -> Dict[str, float]:
    wt_ratio = patient.weight_kg / XU_WT_REF
    pma = float(patient.pma_weeks)
    f_mat = pma ** XU_HILL / (pma ** XU_HILL + XU_TM50_WEEKS ** XU_HILL)
    f_scr = math.exp(-XU_THETA_SCR * (float(patient.scr_umol_l)
                                      - xu_scr_std_umol_l(patient.age_years)))
    return {
        "CL": 7.56 * wt_ratio ** 0.75 * f_mat * f_scr,
        "V1": 85.6 * wt_ratio,
        "Q": 1.30 * wt_ratio ** 0.75,
        "V2": 286.0 * wt_ratio,
    }


XU2026 = PopPKModel(
    name="xu2026",
    label="Xu 2026 (3 months-18 y, renal maturation, Australia-wide)",
    reference=("Xu H, et al. Model-informed precision dosing of vancomycin in "
               "children 3 months to 18 years of age using Australia-wide data. "
               "Antimicrob Agents Chemother. 2026. doi:10.1128/aac.01840-25"),
    typical=_xu_typical,
    eta_params=("CL", "V1"),
    omega=omega_from_cv([0.221, 0.329]),
    sigma_prop=0.233,
    required=("weight_kg", "age_years", "scr_umol_l", "pma_weeks"),
    covariate_ranges={"age_years": (0.25, 18.0)},
    population="children 3 months to 18 years",
    notes=("Two-compartment, allometrically scaled to a 70 kg adult (CL and Q "
           "exponent 0.75; V1 and V2 exponent 1). Clearance carries a sigmoidal "
           "postmenstrual-age maturation function (Hill 3.4, TM50 47.7 weeks, both "
           "fixed from Rhodin et al.) and an exponential effect of serum creatinine "
           "relative to an age-adjusted reference. Serum creatinine is in umol/L. "
           "Between-subject variability on CL (22.1% CV) and V1 (32.9% CV); "
           "residual error proportional, 23.3%. The published model also carries "
           "inter-occasion variability on CL of 18.7% CV, which is not applied here "
           "because a single fit has no occasion structure. Developed at The Royal "
           "Children's Hospital Melbourne (627 children, 3687 concentrations) and "
           "externally validated at three other Australian centres. "
           "NOTE ON THE VOLUMES: this model's central volume (85.6 L/70 kg = "
           "1.22 L/kg) is two to three times that of the other models here, and its "
           "steady-state volume (5.3 L/kg) is far above the 0.4-0.7 L/kg usually "
           "quoted for vancomycin, with a peripheral compartment that equilibrates "
           "very slowly (Q 1.30 L/h). Predicted peaks are correspondingly low and "
           "peak-to-trough fluctuation flat. This does not affect AUC24 or the dose, "
           "which depend only on clearance, and the implementation reproduces the "
           "paper's own simulated troughs; but do not use this model to predict peak "
           "concentrations or to support trough-based decisions."),
)


# ==========================================================================
# Le et al. 2013 -- children >= 3 months, AUC-guided dosing.  One-compartment.
# ==========================================================================
def _le_typical(patient: Patient) -> Dict[str, float]:
    wt = patient.weight_kg
    scr = float(patient.scr_mg_dl)
    age_days = float(patient.age_days)
    return {
        "CL": 0.248 * wt ** 0.75 * (0.48 / scr) ** 0.361
              * (math.log(age_days) / 7.8) ** 0.995,
        "V": 0.636 * wt,
    }


LE2013 = PopPKModel(
    name="le2013",
    label="Le 2013 (>= 3 months, one-compartment, AUC-guided)",
    reference=("Le J, Bradley JS, Murray W, et al. Improved vancomycin dosing in "
               "children using area-under-the-curve exposure. Pediatr Infect Dis J. "
               "2013;32(4):e155-e163. doi:10.1097/INF.0b013e318286378e"),
    typical=_le_typical,
    eta_params=("CL", "V"),
    omega=np.diag([0.125, 0.031]),
    sigma_prop=math.sqrt(0.0857),
    required=("weight_kg", "scr_mg_dl", "age_days"),
    covariate_ranges={"age_years": (0.25, 21.0)},
    population="children >= 3 months (median 6.6 y, 22.7 kg); neonates excluded",
    notes=("One-compartment with first-order elimination: "
           "CL (L/h) = 0.248 * Wt^0.75 * (0.48/SCr)^0.361 * (ln(age_days)/7.8)^0.995 "
           "and V (L) = 0.636 * Wt, with serum creatinine in mg/dL and age in days. "
           "The tabulated random-effect variances are used directly (eta_CL 0.125, "
           "eta_V 0.031, i.e. 35% and 18%); residual error variance 0.0857 gives a "
           "29% proportional error. 702 patients, 1660 concentrations. Because the "
           "distribution phase is not modelled, early post-infusion predictions are "
           "lower than a two-compartment model would give; the published analysis "
           "notes most samples were drawn after distribution."),
)




# ==========================================================================
# Lamarre et al. 2000 -- pediatric patients < 18 years.  Two-compartment.
# ==========================================================================
LAMARRE_A = 0.46             # slope on creatinine clearance (dimensionless)
LAMARRE_B_L_PER_H_PER_KG = 0.3 * 60.0 / 1000.0   # 0.3 mL/min/kg expressed in L/h/kg


def lamarre_creatinine_clearance(patient: Patient) -> float:
    """CCL (L/h): Schwartz creatinine clearance de-normalised to the patient's BSA."""
    clcr_per_173 = schwartz_1976(patient.height_cm, patient.scr_mg_dl,
                                 patient.age_years, patient.sex,
                                 patient.gestational_age_weeks)      # mL/min/1.73 m^2
    bsa = body_surface_area(patient.weight_kg)
    return clcr_per_173 * 0.06 * bsa / 1.73                          # L/h


def _lamarre_typical(patient: Patient) -> Dict[str, float]:
    wt = patient.weight_kg
    ccl = lamarre_creatinine_clearance(patient)
    return {
        "CL": LAMARRE_A * ccl + LAMARRE_B_L_PER_H_PER_KG * wt,
        "V1": 0.27 * wt,
        "Q": 0.16 * wt,
        "V2": 0.16 * wt,
    }


LAMARRE2000 = PopPKModel(
    name="lamarre2000",
    label="Lamarre 2000 (< 18 y, creatinine-clearance slope model)",
    reference=("Lamarre P, Lebel D, Ducharme MP. A population pharmacokinetic model "
               "for vancomycin in pediatric patients and its predictive value in a "
               "naive population. Antimicrob Agents Chemother. 2000;44(2):278-282. "
               "doi:10.1128/AAC.44.2.278-282.2000"),
    typical=_lamarre_typical,
    eta_params=("CL", "V1", "V2"),
    omega=omega_from_cv([0.45, 0.42, 0.43]),
    sigma_prop=0.065,
    required=("weight_kg", "height_cm", "scr_mg_dl", "age_years"),
    covariate_ranges={"age_years": (0.0, 18.0)},
    population="pediatric patients under 18 years (n = 78, single centre)",
    notes=("Two-compartment, fitted by an iterative two-stage method. Clearance is "
           "linear in creatinine clearance: CL = 0.46 * CCL + 0.018 * weight (L/h), "
           "where CCL is the ORIGINAL (1976) Schwartz clearance -- k = 0.45/0.55/0.70 "
           "by age and sex, not the modern bedside k = 0.413 -- converted to L/h and "
           "de-normalised from 1.73 m2 using BSA = (4*BW + 7)/(BW + 90). Age and sex "
           "enter the model only through that k. Volumes scale linearly with weight: "
           "V1 0.27 L/kg, V2 0.16 L/kg, Q 0.16 L/h/kg. "
           "TWO DEVIATIONS FROM THE PUBLICATION, both documented here rather than "
           "hidden: (1) the published between-subject variability of 45% sits on the "
           "slope A, whereas this implementation applies it to the whole of CL, which "
           "differs only through the weight-proportional intercept (about 19% of "
           "typical clearance); (2) the published residual error has an additive floor "
           "tied to the assay limit of detection whose value is not tabulated, so only "
           "the 6.5% proportional part is used. That 6.5% is far tighter than the other "
           "models here, so fits will track the measured levels very closely and shrink "
           "toward the prior very little -- treat individual estimates from this model "
           "with corresponding caution."),
)


# ==========================================================================
# Colin et al. 2019 -- pooled analysis spanning neonates to the elderly.
# Two-compartment with maturation and age-related decline.
# ==========================================================================
COLIN_WT_REF = 70.0
COLIN_PMA50_WEEKS = 46.4
COLIN_GAMMA1 = 2.89
COLIN_AGE50_YEARS = 61.6
COLIN_GAMMA2 = 2.24
COLIN_THETA_SCR = 0.649      # per mg/dL


def colin_scr_std_mg_dl(pma_years: float) -> float:
    """Typical-for-age serum creatinine (mg/dL) as an empiric function of PMA."""
    pma = float(pma_years)
    return math.exp(-1.228 + math.log10(pma) * 0.672
                    + 6.27 * math.exp(-3.11 * pma))


def _colin_typical(patient: Patient) -> Dict[str, float]:
    f_size = patient.weight_kg / COLIN_WT_REF
    pma_wk = float(patient.pma_weeks)
    pma_yr = pma_wk / WEEKS_PER_YEAR
    f_mat = pma_wk ** COLIN_GAMMA1 / (pma_wk ** COLIN_GAMMA1
                                      + COLIN_PMA50_WEEKS ** COLIN_GAMMA1)
    f_decline = (COLIN_AGE50_YEARS ** COLIN_GAMMA2
                 / (COLIN_AGE50_YEARS ** COLIN_GAMMA2 + pma_yr ** COLIN_GAMMA2))
    f_scr = math.exp(-COLIN_THETA_SCR * (float(patient.scr_mg_dl)
                                         - colin_scr_std_mg_dl(pma_yr)))
    return {
        "CL": 5.31 * f_size ** 0.75 * f_mat * f_decline * f_scr,
        "V1": 42.9 * f_size,
        "Q": 3.22 * f_size ** 0.75,
        "V2": 41.7 * f_size,
    }


COLIN2019 = PopPKModel(
    name="colin2019",
    label="Colin 2019 (neonate to elderly, pooled analysis)",
    reference=("Colin PJ, Allegaert K, Thomson AH, et al. Vancomycin pharmacokinetics "
               "throughout life: results from a pooled population analysis and "
               "evaluation of current dosing recommendations. Clin Pharmacokinet. "
               "2019;58(6):767-780. doi:10.1007/s40262-018-0727-5"),
    typical=_colin_typical,
    eta_params=("CL", "V1", "V2"),
    omega=omega_from_cv([0.279, 0.273, 0.979]),
    sigma_prop=0.215,
    sigma_add=1.23,
    required=("weight_kg", "scr_mg_dl", "pma_weeks"),
    population="neonates through the elderly (pooled across 14 studies)",
    notes=("Two-compartment with compartmental allometry: CL = 5.31 * (WGT/70)^0.75 * "
           "Fmat * Fdecline * Fscr, V1 = 42.9 * WGT/70, V2 = 41.7 * WGT/70, "
           "Q2 = 3.22 * (WGT/70)^0.75, all per 70 kg. Fmat is a sigmoid in "
           "postmenstrual age (PMA50 46.4 weeks, gamma 2.89), reaching 90% of adult "
           "clearance by 2 years PMA; Fdecline is a sigmoid decline in PMA years "
           "(AGE50 61.6 y, gamma 2.24); Fscr = exp(-0.649 * (SCr - SCRstd)) with serum "
           "creatinine in mg/dL and SCRstd an empiric function of PMA. Between-subject "
           "variability on CL (27.9% CV), V1 (27.3%) and V2 (97.9%); none on Q2, as "
           "published. Residual error is combined: 21.5% proportional plus 1.23 mg/L "
           "additive. The two study-specific effects in the published model "
           "(haematological malignancy on CL, heel-prick sampling on V1 and Q2) are "
           "not applied, since they describe source cohorts rather than a new patient. "
           "This is the only shipped model that spans neonates and older children in "
           "one prior."),
)


# ==========================================================================
# Frymoyer et al. 2014 -- neonates in a level III NICU.  One-compartment.
# ==========================================================================
FRYMOYER_WT_REF = 2.9        # kg
FRYMOYER_TM50_WEEKS = 34.8
FRYMOYER_HILL = 4.53


def _frymoyer_typical(patient: Patient) -> Dict[str, float]:
    f_size = patient.weight_kg / FRYMOYER_WT_REF
    pma = float(patient.pma_weeks)
    f_mat = 1.0 / (1.0 + (pma / FRYMOYER_TM50_WEEKS) ** (-FRYMOYER_HILL))
    f_scr = (1.0 / float(patient.scr_mg_dl)) ** 0.267
    return {
        "CL": 0.345 * f_size ** 0.75 * f_mat * f_scr,
        "V": 1.75 * f_size,
    }


FRYMOYER2014 = PopPKModel(
    name="frymoyer2014",
    label="Frymoyer 2014 (neonates, AUC-guided)",
    reference=("Frymoyer A, Hersh AL, El-Komy MH, et al. Association between "
               "vancomycin trough concentration and area under the concentration-time "
               "curve in neonates. Antimicrob Agents Chemother. 2014;58(11):6454-6461. "
               "doi:10.1128/AAC.03620-14"),
    typical=_frymoyer_typical,
    eta_params=("CL", "V"),
    omega=omega_from_cv([0.216, 0.109]),
    sigma_prop=0.205,
    sigma_add=1.3,
    required=("weight_kg", "scr_mg_dl", "pma_weeks"),
    covariate_ranges={"pma_weeks": (22.0, 52.0)},
    population=("neonates in a level III NICU (n = 249; median PMA 39 weeks, "
                "IQR 32-42; median weight 2.9 kg, IQR 1.6-3.7)"),
    notes=("One-compartment: CL = 0.345 * (WT/2.9)^0.75 * Fmat * (1/SCr)^0.267 and "
           "V = 1.75 * (WT/2.9), with serum creatinine in mg/dL. Fmat is a sigmoid in "
           "postmenstrual age, 1/(1 + (PMA/34.8)^-4.53). Between-subject variability "
           "on CL (21.6% CV) and V (10.9% CV); residual error combined, 20.5% "
           "proportional plus 1.3 mg/L additive. Developed specifically around the "
           "AUC24 target rather than troughs. The 22-52 week applicability band is "
           "the neonatal population the study enrolled (median PMA 39 weeks, "
           "IQR 32-42), not a covariate range the paper tabulates; it is set so that "
           "applying a neonatal model to an older child raises a warning."),
)


# ==========================================================================
# Holford et al. 2024 -- joint gentamicin/amikacin/vancomycin model, premature
# neonates to adults (NextDose "GAV 2023").  Two-compartment.
#
# Transcribed from the published NM-TRAN control stream (Br J Clin Pharmacol
# 2024;90:1066-80, Supporting Information S1), vancomycin branch (DRUG = 1).
# Fat-free mass, creatinine production rate (CPR) and normal GFR come from
# O'Hanlon et al., CPT Pharmacometrics Syst Pharmacol 2023;12:401-12 (Table 1,
# Equations 5-9 with the 2023 erratum: FFMIN = FFMAT x FMAT_MAX, FFMAT fixed 0).
# ==========================================================================
_LN2 = math.log(2.0)
HOLFORD_FFM_STD = 56.1          # kg, FFM of the 70 kg, 176 cm standard male
HOLFORD_FAT_STD = 13.9          # kg, 70 - 56.1
HOLFORD_ADULT_PMA_WEEKS = 20 * 52 + 40      # 1080
# Jaffe -> enzymatic creatinine: (88.4 - 22.25)/88.4 = 0.748 (O'Hanlon Eq 12)
HOLFORD_FSCR = (88.4 - 22.25) / 88.4

# O'Hanlon 2023 Table 1: fractional FFM (FRFFM) model, sex specific.
_FRFFM = {
    "M": dict(FMAT_PRE=5.20, TFF_PRE=0.060, FMAT_MAX=1.74, A50=3.54,
              HILL_Y=0.127, HILL_O=0.164, CAP=1.16),
    "F": dict(FMAT_PRE=0.837, TFF_PRE=0.12, FMAT_MAX=2.03, A50=1.53,
              HILL_Y=0.0339, HILL_O=0.0306, CAP=1.19),
}

# Holford 2024 $THETA, vancomycin branch (final estimates; the control stream's
# initial values reproduce the paper's Table 2).
_HOLFORD_V = dict(
    FGFR=0.567, CLGFR_EMAX=1.0, RF50=0.531, HILL_LT=1.88, HILL_GE=5.08,
    CLNGFR=1.67, CLNGFR_TM50=57.8, CLNGFR_HILL=3.24, CLNGFR_PNAMAX=0.962,
    CLNGFR_PNAT50=2.84, CLNGFR_FFAT=0.324,
    V=32.5, V_FFAT=0.337, VP=27.4, VP_FFAT=11.7, Q=1.33, Q_FFAT=4.02,
    PWR_CL=0.75, PWR_Q=0.75, PWR_V1=1.0, PWR_V2=1.0,
    FNEOVC=0.109, T2NEOVC=266.0, FNEOVP=0.0156, T2NEOVP=247.0,   # half-lives in days
    # normal GFR (O'Hanlon 2023 Table 3 as coded in the control stream)
    STDGFR=116 * 60 / 1000, TM50GFR=33.6, HILLGFR=3.49, PNAMAX=0.588, PNAT50=6.94,
    # creatinine production (Matthews 2004 / O'Hanlon 2023 Table 2)
    STDCPR_JAFFE=0.516, CPR_FSDY4=0.86, CPR_INT={"M": 0.764, "F": 0.72},
    CPR_SLOPE={"M": -0.0315, "F": -0.0274},
)


def _sex_code(sex: Optional[str]) -> Optional[str]:
    if sex is None:
        return None
    s = str(sex).strip().lower()
    if s in ("m", "male", "boy", "1"):
        return "M"
    if s in ("f", "female", "girl", "2"):
        return "F"
    return None


def janmahasatian_ffm(weight_kg: float, height_cm: float, sex: str) -> float:
    """Adult fat-free mass (Janmahasatian 2005), kg."""
    bmi = weight_kg / (height_cm / 100.0) ** 2
    if _sex_code(sex) == "F":
        return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)
    return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)


def ohanlon_frffm(pma_weeks: float, sex: str) -> float:
    """Fraction of adult-predicted FFM (O'Hanlon 2023 Eqs 6-9, with erratum)."""
    p = _FRFFM[_sex_code(sex)]
    pma_y = pma_weeks / 52.0
    ffmin = 0.0 * p["FMAT_MAX"]                                   # FFMAT fixed at 0
    ffneo = (p["FMAT_PRE"] - ffmin) * math.exp(-_LN2 / p["TFF_PRE"] * (pma_y - 24.0 / 52.0))
    hill = p["HILL_Y"] if pma_y < p["A50"] else p["HILL_O"]
    ffkid = (p["FMAT_MAX"] - ffmin) / (1.0 + (pma_y / p["A50"]) ** (-hill))
    return min(ffmin + ffneo + ffkid, p["CAP"])


def ohanlon_ffm(weight_kg: float, height_cm: float, sex: str, pma_weeks: float) -> float:
    """Fat-free mass from neonates to adults (O'Hanlon 2023), kg."""
    return ohanlon_frffm(pma_weeks, sex) * janmahasatian_ffm(weight_kg, height_cm, sex)


def holford_renal_function(patient: Patient, scr_assay: str = "enzymatic"
                           ) -> Dict[str, float]:
    """Renal function RF = eGFR / normal GFR, as coded in the Holford 2024 $PK.

    ``scr_assay`` is the method behind the patient's serum creatinine:
    "enzymatic" (also IDMS-traceable compensated Jaffe) or "jaffe" (uncompensated;
    multiplied by 0.748 to its enzymatic equivalent, as for most source studies).
    Serum creatinine is treated as being at steady state.
    """
    h = _HOLFORD_V
    sex = _sex_code(patient.sex)
    if sex is None:
        raise ValueError("Model 'holford2024' needs the patient's sex (M or F): fat-free "
                         "mass and creatinine production are sex specific.")
    for need in ("height_cm", "scr_umol_l", "pma_weeks", "age_days"):
        if patient.get(need) is None:
            raise ValueError(f"Model 'holford2024' requires {need}.")
    wt = float(patient.weight_kg)
    pma = float(patient.pma_weeks)
    pnad = float(patient.age_days)
    pnay = pnad / DAYS_PER_YEAR
    ffm = ohanlon_ffm(wt, float(patient.height_cm), sex, pma)
    ffmi = min(ffm, 0.95 * wt)                                        # FFMKGI

    # normal GFR (FFTGFR = 0, so size is FFM alone)
    fsz_gfr = (ffmi / HOLFORD_FFM_STD) ** 0.75
    fmt_gfr = 1.0 / (1.0 + (pma / h["TM50GFR"]) ** (-h["HILLGFR"]))
    fpnt_gfr = 1.0 - h["PNAMAX"] + h["PNAMAX"] * (1.0 - math.exp(-_LN2 / h["PNAT50"] * pnad))
    ngfr = h["STDGFR"] * fsz_gfr * fmt_gfr * fpnt_gfr                  # L/h

    j2enz = 1.0 if scr_assay == "enzymatic" else HOLFORD_FSCR
    scr_u = float(patient.scr_umol_l)
    scrz = scr_u * j2enz
    # creatinine production rate
    if pma < HOLFORD_ADULT_PMA_WEEKS:
        fpma_cpr = (h["CPR_INT"][sex] if pma <= 44.0
                    else 1.0 + h["CPR_SLOPE"][sex] / 100.0 * (HOLFORD_ADULT_PMA_WEEKS - pma))
    else:
        fpma_cpr = 1.0
    if pnay >= 20.0:                                   # Matthews et al. 2004, adults
        if scrz >= 60.0 * j2enz:
            fsex = 0.82 if sex == "F" else 1.0
            fmt_cpr = (112.0 - pnay) * fsex / (112.0 - 40.0)
        else:
            fmt_cpr = (119.0 - pnay) * 0.7 / (119.0 - 40.0)
    else:
        fmt_cpr = 1.0
    stdcprz = h["STDCPR_JAFFE"] * HOLFORD_FSCR                         # enzymatic
    cpr = stdcprz * fmt_cpr * fpma_cpr * h["CPR_FSDY4"] * (ffmi / HOLFORD_FFM_STD)  # mmol/h
    if pnad < 2.0:                                     # creatinine still maternal
        egfr, rf = float("nan"), 1.0
    else:
        egfr = cpr / (scrz / 1000.0)                                   # L/h
        rf = egfr / ngfr
    return {"ffm_kg": ffm, "ffm_used_kg": ffmi, "ngfr_l_h": ngfr, "cpr_mmol_h": cpr,
            "egfr_l_h": egfr, "rf": rf}


def _holford_components(patient: Patient, scr_assay: str) -> Dict[str, float]:
    h = _HOLFORD_V
    r = holford_renal_function(patient, scr_assay)
    wt = float(patient.weight_kg)
    ffmi = r["ffm_used_kg"]
    pma = float(patient.pma_weeks)
    pnad = float(patient.age_days)
    rf = r["rf"]

    def fsize(ffat: float, pwr: float) -> float:
        nfm = ffmi + ffat * (wt - ffmi)
        nfm_std = HOLFORD_FFM_STD + ffat * HOLFORD_FAT_STD
        return (nfm / nfm_std) ** pwr

    # GFR clearance: asymmetric sigmoid in RF
    hill = h["HILL_LT"] if rf <= h["RF50"] else h["HILL_GE"]
    f_rf = h["CLGFR_EMAX"] / (1.0 + (rf / h["RF50"]) ** (-hill))
    cl_gfr = r["ngfr_l_h"] * h["FGFR"] * f_rf
    # non-GFR clearance: proportional to RF, own maturation and size
    fpma = 1.0 / (1.0 + (pma / h["CLNGFR_TM50"]) ** (-h["CLNGFR_HILL"]))
    fpnt = (1.0 if pnad <= 0 else
            1.0 - h["CLNGFR_PNAMAX"]
            + h["CLNGFR_PNAMAX"] * (1.0 - math.exp(-_LN2 / h["CLNGFR_PNAT50"] * pnad)))
    cl_ngfr = (h["CLNGFR"] * rf * fpma * fpnt
               * fsize(h["CLNGFR_FFAT"], h["PWR_CL"]))   # FVNT = 1; NSAIDs not modelled
    v1 = (h["V"] * (1.0 + h["FNEOVC"] * math.exp(-_LN2 / h["T2NEOVC"] * pnad))
          * fsize(h["V_FFAT"], h["PWR_V1"]))             # inotropes (x1.15) not modelled
    v2 = (h["VP"] * (1.0 + h["FNEOVP"] * math.exp(-_LN2 / h["T2NEOVP"] * pnad))
          * fsize(h["VP_FFAT"], h["PWR_V2"]))
    q = h["Q"] * fsize(h["Q_FFAT"], h["PWR_Q"])
    return {"CL_GFR": cl_gfr, "CL_NGFR": cl_ngfr, "V1": v1, "Q": q, "V2": v2, **r}


def _holford_individual(scr_assay: str):
    def individual(patient: Patient, eta: Optional[np.ndarray]) -> Dict[str, float]:
        c = _holford_components(patient, scr_assay)
        e = np.zeros(5) if eta is None else eta
        cl = c["CL_GFR"] * math.exp(e[0]) + c["CL_NGFR"] * math.exp(e[1])
        return {"CL": cl, "V1": c["V1"] * math.exp(e[2]),
                "Q": c["Q"] * math.exp(e[3]), "V2": c["V2"] * math.exp(e[4])}
    return individual


def make_holford2024(scr_assay: str = "enzymatic") -> PopPKModel:
    """Holford 2024 for a given creatinine assay ("enzymatic" or "jaffe")."""
    if scr_assay not in ("enzymatic", "jaffe"):
        raise ValueError("scr_assay must be 'enzymatic' or 'jaffe'.")
    ind = _holford_individual(scr_assay)
    suffix = "" if scr_assay == "enzymatic" else "_jaffe"
    assay_text = ("enzymatic / IDMS-traceable creatinine" if scr_assay == "enzymatic"
                  else "uncompensated Jaffe creatinine (x0.748 to enzymatic)")
    return PopPKModel(
        name=f"holford2024{suffix}",
        label=f"Holford 2024 (neonates to adults, GFR-based; {assay_text})",
        reference=("Holford N, O'Hanlon CJ, Allegaert K, et al. A physiological approach "
                   "to renal clearance: from premature neonates to adults. Br J Clin "
                   "Pharmacol. 2024;90(4):1066-1080. doi:10.1111/bcp.15978; with "
                   "O'Hanlon CJ, Holford N, Sumpter A, Al-Sallami HS. CPT Pharmacometrics "
                   "Syst Pharmacol. 2023;12:401-412. doi:10.1002/psp4.12924"),
        typical=lambda p: ind(p, None),
        individual_fn=ind,
        eta_params=("CL_GFR", "CL_NGFR", "V1", "Q", "V2"),
        # $OMEGA variances, vancomycin: BSV_CLGFR_VANC, BSV_CLNGFR, BSV_V, PPV_Q, PPV_VP
        omega=np.diag([0.049, 0.134, 0.00039, 0.0645, 0.182]),
        # $ERROR: SD = FRVSDY*sqrt((0.15*CP)^2 + 2.14^2)*exp(ETA9); reference study
        # (FRVSDY = 1) and the typical individual (ETA9 = 0) are used.
        sigma_prop=0.15,
        sigma_add=2.14,
        required=("weight_kg", "height_cm", "scr_umol_l", "sex", "pma_weeks", "age_days"),
        covariate_ranges={"age_years": (0.0, 100.0)},
        population=("pooled GAVamycin data: 18 studies, 9,901 patients on gentamicin, "
                    "amikacin or vancomycin, premature neonates to elderly adults; "
                    f"assumes {assay_text}"),
        notes=("Two-compartment. CL = CL_GFR + CL_NGFR. CL_GFR = 0.567 x normal GFR x "
               "sigmoid(RF; RF50 0.531, Hill 1.88 below / 5.08 above). CL_NGFR = 1.67 L/h "
               "x RF x maturation x (NFM/NFMstd)^0.75 with Ffat 0.324. RF = eGFR/normal "
               "GFR, eGFR = creatinine production rate / SCr, all scaled by fat-free mass "
               "(O'Hanlon 2023). V1 32.5, V2 27.4 L, Q 1.33 L/h per 70 kg standard, "
               "normal-fat-mass scaled (Ffat 0.337, 11.7, 4.02). Separate log-normal BSV "
               "on CL_GFR (omega^2 0.049) and CL_NGFR (0.134), V1 (0.00039), Q (0.0645), "
               "V2 (0.182), variances taken directly from $OMEGA. Residual error 15% + "
               "2.14 mg/L for the reference study; the study-specific multipliers "
               "(0.61-3.57) and the between-subject variability of residual error "
               "(omega^2 0.629) are not used. Not modelled: inotropes (V x1.15), NSAIDs "
               "(ibuprofen CL_NGFR x0.744), non-steady-state creatinine kinetics, "
               "between-occasion variability (negligible). Requires sex, height and "
               "postmenstrual age (term assumed when gestational age is not given)."),
    )


HOLFORD2024 = make_holford2024("enzymatic")
HOLFORD2024_JAFFE = make_holford2024("jaffe")


MODELS: Dict[str, PopPKModel] = {m.name: m for m in (
    SMIT2021, XU2026, LE2013, LAMARRE2000, COLIN2019, FRYMOYER2014,
    HOLFORD2024, HOLFORD2024_JAFFE)}
DEFAULT_MODEL = "smit2021"


def get_model(name: str | PopPKModel = DEFAULT_MODEL) -> PopPKModel:
    if isinstance(name, PopPKModel):
        return name
    try:
        return MODELS[str(name).lower()]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODELS)}") from None


def register_model(model: PopPKModel) -> PopPKModel:
    """Add a model to the registry so it can be selected by name."""
    MODELS[model.name] = model
    return model


def describe_models() -> str:
    lines = []
    for m in MODELS.values():
        lines.append(f"{m.name:10s} {m.n_compartments}-cpt  {m.label}")
        lines.append(f"{'':10s} needs: {', '.join(m.required)}")
    return "\n".join(lines)
