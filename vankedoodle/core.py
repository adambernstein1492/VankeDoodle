"""Core data types for VankeDoodle: patients, doses, levels, regimens.

All internal time is expressed in hours relative to a single time origin
(normally the start of the first vancomycin dose).  Users may supply times
either as plain numbers (hours) or as datetimes / ISO-8601 strings; the
helpers here normalise both to hours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional, Sequence, Union

TimeLike = Union[float, int, str, datetime]


def _parse_dt(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date/time: {value!r}")


def to_hours(value: TimeLike, origin: Optional[datetime] = None) -> float:
    """Convert a time-like value to hours after ``origin``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    dt = _parse_dt(value)
    if origin is None:
        raise ValueError("A datetime was supplied but no time origin is set.")
    return (dt - origin).total_seconds() / 3600.0


DateLike = Union[str, date, datetime]


def _parse_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _parse_dt(value).date()


def age_from_birthdate(birthdate: DateLike, on: Optional[DateLike] = None
                       ) -> dict:
    """Postnatal age on the date ``on`` (default: today), in days and years.

    Returns ``{"age_days": int, "age_years": float}``; years use 365.25 days.
    Raises ValueError when the birthdate falls after ``on``.
    """
    born = _parse_date(birthdate)
    ref = _parse_date(on) if on is not None else date.today()
    days = (ref - born).days
    if days < 0:
        raise ValueError(f"The birthdate {born.isoformat()} is after "
                         f"{ref.isoformat()}, the date the age is taken on.")
    return {"age_days": days, "age_years": days / DAYS_PER_YEAR}


def bedside_schwartz(height_cm: float, scr_mg_dl: float) -> float:
    """Bedside Schwartz creatinine clearance, mL/min/1.73 m^2.

    CLcr = 0.413 * height(cm) / serum creatinine(mg/dL).
    """
    if height_cm is None or scr_mg_dl is None:
        raise ValueError("Height and serum creatinine are required for bedside Schwartz.")
    if scr_mg_dl <= 0:
        raise ValueError("Serum creatinine must be > 0 mg/dL.")
    return 0.413 * float(height_cm) / float(scr_mg_dl)


SCR_MG_DL_PER_UMOL_L = 1.0 / 88.4   # creatinine: 1 mg/dL = 88.4 umol/L
WEEKS_PER_YEAR = 52.1775
DAYS_PER_YEAR = 365.25
TERM_GESTATION_WEEKS = 40.0


def body_surface_area(weight_kg: float) -> float:
    """Body surface area (m^2) from weight, BSA = (4*BW + 7) / (BW + 90).

    The weight-only formula used by Lamarre et al. 2000.
    """
    w = float(weight_kg)
    return (4.0 * w + 7.0) / (w + 90.0)


SCHWARTZ_K_PRETERM_INFANT = 0.33     # < 1 year, born preterm
SCHWARTZ_K_TERM_INFANT = 0.45        # < 1 year, born at term
SCHWARTZ_K_CHILD = 0.55              # 1-12 y, and adolescent girls
SCHWARTZ_K_ADOLESCENT_MALE = 0.70    # 13-18 y, boys


def schwartz_k(age_years: Optional[float], sex: Optional[str] = None,
               gestational_age_weeks: Optional[float] = None) -> float:
    """The age/sex-dependent constant of the original (1976) Schwartz formula."""
    if age_years is not None and age_years < 1.0:
        preterm = gestational_age_weeks is not None and gestational_age_weeks < 37.0
        return SCHWARTZ_K_PRETERM_INFANT if preterm else SCHWARTZ_K_TERM_INFANT
    if age_years is not None and age_years >= 13.0:
        if sex is not None and str(sex).strip().lower() in ("m", "male", "boy"):
            return SCHWARTZ_K_ADOLESCENT_MALE
        return SCHWARTZ_K_CHILD          # girls, and unknown sex (the lower value)
    return SCHWARTZ_K_CHILD


def schwartz_1976(height_cm: float, scr_mg_dl: float, age_years: Optional[float] = None,
                  sex: Optional[str] = None,
                  gestational_age_weeks: Optional[float] = None) -> float:
    """Original Schwartz creatinine clearance, mL/min/1.73 m^2.

    CLcr = k * height(cm) / serum creatinine(mg/dL), with k from
    :func:`schwartz_k`.  This is the formula in use when the older models were
    built; :func:`bedside_schwartz` (k = 0.413) is its modern recalibration for
    enzymatic creatinine assays.  They are not interchangeable -- a model is
    implemented with whichever its authors used.
    """
    k = schwartz_k(age_years, sex, gestational_age_weeks)
    if scr_mg_dl <= 0:
        raise ValueError("Serum creatinine must be > 0 mg/dL.")
    return k * float(height_cm) / float(scr_mg_dl)


@dataclass
class Patient:
    """Patient covariates.

    Which fields are required depends on the population model chosen; each model
    declares its own requirements and checks them (see
    :meth:`vankedoodle.models.PopPKModel.check`).  Supply what the chart gives
    you — weight, age, height, serum creatinine — and, for neonates and infants,
    gestational age and postnatal age so that postmenstrual age can be formed.

    Serum creatinine may be given in either unit; the other is derived.
    """
    weight_kg: float
    age_years: Optional[float] = None
    height_cm: Optional[float] = None
    scr_mg_dl: Optional[float] = None
    scr_umol_l: Optional[float] = None
    crcl: Optional[float] = None               # mL/min/1.73 m^2, bedside Schwartz
    gestational_age_weeks: Optional[float] = None
    postnatal_age_days: Optional[float] = None
    sex: Optional[str] = None
    patient_id: Optional[str] = None
    birthdate: Optional[DateLike] = None       # sets the two ages above when they are absent
    age_on: Optional[DateLike] = None          # the date the age is taken on (default: today)
    assumed_term_gestation: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.weight_kg is None or self.weight_kg <= 0:
            raise ValueError("weight_kg must be > 0.")
        if self.scr_mg_dl is None and self.scr_umol_l is not None:
            self.scr_mg_dl = float(self.scr_umol_l) * SCR_MG_DL_PER_UMOL_L
        elif self.scr_umol_l is None and self.scr_mg_dl is not None:
            self.scr_umol_l = float(self.scr_mg_dl) / SCR_MG_DL_PER_UMOL_L
        if self.scr_mg_dl is not None and self.scr_mg_dl <= 0:
            raise ValueError("Serum creatinine must be > 0.")
        if self.birthdate is not None:
            self.birthdate = _parse_date(self.birthdate)
            self.age_on = _parse_date(self.age_on) if self.age_on is not None else date.today()
            age = age_from_birthdate(self.birthdate, self.age_on)
            if self.postnatal_age_days is None:
                self.postnatal_age_days = float(age["age_days"])
            if self.age_years is None:
                self.age_years = age["age_years"]
        if self.age_years is None and self.postnatal_age_days is not None:
            self.age_years = float(self.postnatal_age_days) / DAYS_PER_YEAR
        if self.crcl is None and self.height_cm is not None and self.scr_mg_dl is not None:
            self.crcl = bedside_schwartz(self.height_cm, self.scr_mg_dl)
        if self.crcl is not None:
            self.crcl = float(self.crcl)

    # -- derived covariates --------------------------------------------
    @property
    def age_days(self) -> Optional[float]:
        if self.postnatal_age_days is not None:
            return float(self.postnatal_age_days)
        if self.age_years is None:
            return None
        return float(self.age_years) * DAYS_PER_YEAR

    @property
    def pma_weeks(self) -> Optional[float]:
        """Postmenstrual age = gestational age + postnatal age.

        When gestational age is not supplied, term gestation (40 weeks) is
        assumed and ``assumed_term_gestation`` is set so the model can warn.
        """
        pna_weeks = None
        if self.postnatal_age_days is not None:
            pna_weeks = float(self.postnatal_age_days) / 7.0
        elif self.age_years is not None:
            pna_weeks = float(self.age_years) * WEEKS_PER_YEAR
        if pna_weeks is None:
            return None
        ga = self.gestational_age_weeks
        if ga is None:
            ga = TERM_GESTATION_WEEKS
            object.__setattr__(self, "assumed_term_gestation", True)
        return float(ga) + pna_weeks

    @property
    def bmi(self) -> Optional[float]:
        if self.height_cm is None:
            return None
        return self.weight_kg / (self.height_cm / 100.0) ** 2

    def get(self, name: str):
        """Fetch a covariate by name, including derived ones."""
        return getattr(self, name, None)

    def describe(self) -> str:
        bits = [f"{self.weight_kg:g} kg"]
        if self.age_years is not None:
            bits.append(f"{self.age_years:.2g} y" if self.age_years >= 1
                        else f"{self.age_days:.0f} d")
        if self.scr_mg_dl is not None:
            bits.append(f"SCr {self.scr_mg_dl:.2f} mg/dL")
        if self.crcl is not None:
            bits.append(f"CLcr {self.crcl:.0f} mL/min/1.73 m2")
        return ", ".join(bits)


@dataclass
class Dose:
    """A single intravenous vancomycin dose given as a zero-order infusion."""
    amount_mg: float
    start_h: float
    duration_h: float = 1.0

    def __post_init__(self):
        if self.amount_mg < 0:
            raise ValueError("Dose amount must be >= 0 mg.")
        if self.duration_h < 0:
            raise ValueError("Infusion duration must be >= 0 h.")

    @property
    def rate_mg_per_h(self) -> float:
        if self.duration_h == 0:
            return float("inf")
        return self.amount_mg / self.duration_h

    @property
    def end_h(self) -> float:
        return self.start_h + self.duration_h


@dataclass
class Level:
    """A measured vancomycin serum concentration."""
    time_h: float
    conc_mg_l: float

    def __post_init__(self):
        if self.conc_mg_l <= 0:
            raise ValueError(
                "Measured concentrations must be > 0 mg/L (the residual error model is "
                "proportional). Drop or impute below-limit samples before fitting."
            )


@dataclass
class DosingHistory:
    """A complete dosing history plus the time origin used for datetimes."""
    doses: list[Dose] = field(default_factory=list)
    origin: Optional[datetime] = None

    # ---- constructors -------------------------------------------------
    @classmethod
    def repeating(cls, dose_mg: float, interval_h: float, n_doses: int,
                  infusion_h: float = 1.0, start: TimeLike = 0.0,
                  first_dose_mg: Optional[float] = None) -> "DosingHistory":
        """The common case: the same dose every ``interval_h`` hours.

        ``first_dose_mg`` allows a loading dose that differs from the rest.
        """
        if n_doses < 1:
            raise ValueError("n_doses must be >= 1.")
        origin = None if isinstance(start, (int, float)) else _parse_dt(start)
        t0 = 0.0 if origin is not None else float(start)
        doses = []
        for i in range(int(n_doses)):
            amt = first_dose_mg if (i == 0 and first_dose_mg is not None) else dose_mg
            doses.append(Dose(amount_mg=amt, start_h=t0 + i * interval_h,
                              duration_h=infusion_h))
        return cls(doses=doses, origin=origin)

    @classmethod
    def from_records(cls, records: Iterable[dict], origin: Optional[TimeLike] = None
                     ) -> "DosingHistory":
        """Build from dicts with keys ``amount_mg``/``dose_mg``, ``time``/``start``
        and optional ``duration_h``/``infusion_h``.  Irregular and missed doses are
        handled naturally because every dose is listed explicitly."""
        recs = list(records)
        org = _parse_dt(origin) if isinstance(origin, (str, datetime)) else None
        if org is None:
            stamps = [r.get("time", r.get("start")) for r in recs]
            dts = [s for s in stamps if isinstance(s, (str, datetime))]
            if dts:
                org = min(_parse_dt(s) for s in dts)
        doses = []
        for r in recs:
            amt = r.get("amount_mg", r.get("dose_mg"))
            t = r.get("time", r.get("start"))
            dur = r.get("duration_h", r.get("infusion_h", 1.0))
            doses.append(Dose(amount_mg=float(amt), start_h=to_hours(t, org),
                              duration_h=float(dur)))
        doses.sort(key=lambda d: d.start_h)
        return cls(doses=doses, origin=org)

    # ---- helpers ------------------------------------------------------
    def add(self, amount_mg: float, time: TimeLike, duration_h: float = 1.0) -> "DosingHistory":
        self.doses.append(Dose(amount_mg=float(amount_mg),
                               start_h=to_hours(time, self.origin),
                               duration_h=float(duration_h)))
        self.doses.sort(key=lambda d: d.start_h)
        return self

    def hours(self, value: TimeLike) -> float:
        return to_hours(value, self.origin)

    def levels(self, records: Iterable[Union[dict, Sequence]]) -> list[Level]:
        """Convert measured levels given as dicts or (time, conc) pairs."""
        out = []
        for r in records:
            if isinstance(r, dict):
                t = r.get("time", r.get("time_h"))
                c = r.get("conc_mg_l", r.get("conc", r.get("level")))
            else:
                t, c = r
            out.append(Level(time_h=self.hours(t), conc_mg_l=float(c)))
        out.sort(key=lambda l: l.time_h)
        return out

    @property
    def last_dose(self) -> Dose:
        return max(self.doses, key=lambda d: d.start_h)

    def total_dose_mg(self) -> float:
        return sum(d.amount_mg for d in self.doses)
