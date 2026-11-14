"""IGNIS NOVA core: curated data, scoring, evidence-readiness, gap analysis and the SI engines.

No Streamlit / Plotly imports here, so everything in this file can be unit-tested on its own.
Data come from the NASA Space Apps challenge brief and the NASA "Combustion Science" page.
Scores (0-5) are analyst-assigned relevance ratings, documented in the Method tab of the app.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Criteria, profiles, statuses
# --------------------------------------------------------------------------------------
CRITERIA = {
    "prevention": ("Prevention (flammability)", "Informs ignition control, flammability limits and material selection."),
    "detection": ("Detection", "Informs smoke / aerosol / dust detection in crewed spaces."),
    "suppression": ("Suppression", "Informs extinguishment (extinction limits, suppressants)."),
    "highO2": ("High-O₂ relevance", "Covers or informs oxygen-enriched atmospheres (Moon base target: 30% O₂ or more)."),
    "modeling": ("Modeling value", "Controlled, structured data suited to predictive flammability / fire-scenario models."),
    "crewEnv": ("Crew-habitat relevance", "Directly relevant to fires in spacecraft, stations, bases and pressurized rovers."),
    "partialG": ("Partial-g relevance", "Informs lunar (1/6 g) or Martian gravity, directly or as an analog."),
}
CRIT_KEYS = list(CRITERIA)

PROFILES = {
    "spacecraft": {
        "name": "Spacecraft / space station",
        "desc": "ISS-like cabins: balanced fire safety, direct crew-habitat relevance.",
        "w": {"prevention": 3, "detection": 3, "suppression": 3, "highO2": 1, "modeling": 2, "crewEnv": 4, "partialG": 0},
    },
    "moon": {
        "name": "Moon base / pressurized rover",
        "desc": "Expected 30% O₂ or more at 10.2 psi or less, plus 1/6 g: oxygen and gravity relevance weigh heavily.",
        "w": {"prevention": 4, "detection": 3, "suppression": 3, "highO2": 5, "modeling": 2, "crewEnv": 4, "partialG": 4},
    },
    "modeling": {
        "name": "Predictive modeling",
        "desc": "Datasets best suited to build and validate flammability / fire-scenario models.",
        "w": {"prevention": 2, "detection": 1, "suppression": 1, "highO2": 2, "modeling": 5, "crewEnv": 2, "partialG": 2},
    },
    "controls": {
        "name": "Next-gen safety controls",
        "desc": "Detection and suppression hardware and strategy.",
        "w": {"prevention": 2, "detection": 5, "suppression": 5, "highO2": 2, "modeling": 1, "crewEnv": 3, "partialG": 0},
    },
}

STATUS = {
    "archived": {"label": "In PSI", "color": "#3fc79a"},
    "current": {"label": "Current, PSI to confirm", "color": "#6da7ec"},
    "planned": {"label": "Planned, not in PSI yet", "color": "#93a0b8"},
}

GRAVITY_LABEL = {
    "micro": "Microgravity",
    "partial": "Partial gravity (lunar)",
    "analog": "Microgravity + partial-g analog",
}

PILLAR_COLOR = {"Prevention": "#3987e5", "Detection": "#199e70", "Suppression": "#d95926"}

# --------------------------------------------------------------------------------------
# Curated records. s = (prevention, detection, suppression, highO2, modeling, crewEnv, partialG)
# --------------------------------------------------------------------------------------
_BRIEF = "Challenge brief"
_PAGE = "NASA Combustion Science page"

RECORDS = [
    dict(id="BASS", name="BASS", full="Burning and Suppression of Solids", status="archived", fuel="Solid",
         pillar="Prevention", facility_group="MSG · SPICE flow duct",
         hardware="Microgravity Science Glovebox (MSG) with the modular fan-driven flow duct built for SPICE",
         fuels="Wide variety of solid fuel samples (fabrics, plastics)",
         conditions="Controlled O₂ concentration and airflow speed", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Burning and extinction characteristics of solid samples",
         objective="Examine the burning and extinction characteristics of a wide variety of fuel samples in microgravity.",
         variables="Fuel sample type; O₂ concentration and airflow speed are the controlling variables for flammability in microgravity.",
         earth="Flammability-based material selection for textiles, buildings and aircraft interiors", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(5, 0, 3, 3, 4, 5, 1)),
    dict(id="BASS-II", name="BASS-II", full="Burning and Suppression of Solids II", status="archived", fuel="Solid",
         pillar="Prevention", facility_group="MSG · SPICE flow duct",
         hardware="Microgravity Science Glovebox (MSG) with the modular fan-driven flow duct built for SPICE",
         fuels="Solid fuel samples (fabrics, plastics)",
         conditions="Controlled O₂ concentration and airflow speed", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Burning and extinction characteristics of solid samples",
         objective="Follow-on to BASS: flammability and extinction of solid materials in microgravity.",
         variables="Fuel sample type; O₂ concentration and airflow speed.",
         earth="Flammability-based material selection for textiles, buildings and aircraft interiors", earth_cited=False,
         source=_BRIEF, s=(5, 0, 3, 3, 4, 5, 1)),
    dict(id="Saffire", name="Saffire-I/II/III", full="Spacecraft Fire Safety Demonstration", status="archived",
         fuel="Solid", pillar="Prevention", facility_group="Spacecraft-scale fire demonstration",
         hardware="Spacecraft-scale fire demonstration (details to confirm in PSI)",
         fuels="Solid materials such as fabrics and plastics",
         conditions="Spacecraft-relevant atmosphere and flow (confirm in PSI)", o2_max=np.nan, gravity_regime="micro", n_exp=3,
         measurements="Flame growth and spread of solid materials (confirm in PSI)",
         objective="Demonstrate fire behavior of spacecraft materials at larger scale to support crew safety.",
         variables="Material and flow conditions per flight (confirm in PSI).",
         earth="Fire-spread models for enclosed spaces", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(5, 1, 0, 3, 5, 5, 1)),
    dict(id="SAME", name="SAME", full="Smoke Aerosol Measurement Experiment", status="archived", fuel="Aerosol / smoke",
         pillar="Detection", facility_group="Aerosol instrument (modified P-Trak)",
         hardware="Smoke and aerosol measurement instrument (modified P-Trak is a key component)",
         fuels="Smoke from heated / burning materials",
         conditions="Different smoke types: hydrocarbon fuels give soot, plastics give recondensed polymer droplets",
         o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Smoke / aerosol particle concentration and size",
         objective="Characterize smoke so spacecraft detectors can detect different types of smoke.",
         variables="Smoke source material (soot-type vs droplet-type smoke).",
         earth="Smoke detectors that discriminate smoke from dust reduce nuisance alarms (NASA spin-off)", earth_cited=True,
         source=_BRIEF + "; " + _PAGE, s=(0, 5, 0, 1, 3, 5, 0)),
    dict(id="SAME-R", name="SAME-R", full="Smoke Aerosol Measurement Experiment - Reflight", status="archived",
         fuel="Aerosol / smoke", pillar="Detection", facility_group="Aerosol instrument (modified P-Trak)",
         hardware="Smoke and aerosol measurement instrument",
         fuels="Smoke from heated / burning materials", conditions="Smoke types relevant to spacecraft detectors",
         o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Smoke / aerosol particle concentration and size",
         objective="Reflight of SAME to extend smoke-detection evidence.",
         variables="Smoke source material.",
         earth="Smoke detectors that discriminate smoke from dust reduce nuisance alarms (NASA spin-off)", earth_cited=True,
         source=_BRIEF, s=(0, 5, 0, 1, 3, 5, 0)),
    dict(id="DAFT", name="DAFT / DAFT-2", full="Dust and Aerosol Measurement Feasibility Test", status="archived",
         fuel="Aerosol / smoke", pillar="Detection", facility_group="Aerosol instrument (modified P-Trak)",
         hardware="SAME smoke instrument reused to detect dust (modified P-Trak)",
         fuels="Dust particles", conditions="Microgravity cabin conditions", o2_max=np.nan, gravity_regime="micro", n_exp=2,
         measurements="Dust / aerosol concentration",
         objective="Ensure the modified P-Trak performs properly in microgravity; the instrument was used for dust detection.",
         variables="Dust vs smoke aerosol.",
         earth="Dust-vs-smoke discrimination for fewer nuisance alarms, e.g. mining (NASA spin-off)", earth_cited=True,
         source=_BRIEF + "; " + _PAGE, s=(0, 3, 0, 0, 1, 3, 0)),
    dict(id="FLEX", name="FLEX", full="Flame Extinguishment Experiment", status="archived", fuel="Liquid droplet",
         pillar="Suppression", facility_group="CIR · MDCA droplet apparatus",
         hardware="Multi-user Droplet Combustion Apparatus (MDCA) in the Combustion Integrated Rack (CIR)",
         fuels="Burning liquid fuel droplets", conditions="Chamber atmosphere diluted with CO₂ or helium",
         o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Extinction limits vs diluent; limiting oxygen index for steady combustion",
         objective="Quantify the effectiveness of inert-gas suppressants and obtain the most conservative limiting oxygen index.",
         variables="Diluent type (CO₂ vs helium) and concentration.",
         earth="Inert-gas suppression strategy for enclosed spaces", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(2, 0, 5, 3, 4, 3, 0)),
    dict(id="FLEX-2", name="FLEX-2", full="Flame Extinguishment Experiment 2", status="archived", fuel="Liquid droplet",
         pillar="Suppression", facility_group="CIR · MDCA droplet apparatus",
         hardware="MDCA in CIR", fuels="Burning liquid fuel droplets",
         conditions="Atmosphere diluted with CO₂ or helium", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Extinction limits vs diluent",
         objective="Follow-on study of flame extinguishment by dilution.",
         variables="Diluent type and concentration.",
         earth="Inert-gas suppression strategy for enclosed spaces", earth_cited=False,
         source=_BRIEF, s=(2, 0, 5, 3, 4, 3, 0)),
    dict(id="CFI", name="CFI", full="Cool Flames Investigation", status="archived", fuel="Liquid droplet",
         pillar="Prevention", facility_group="CIR · MDCA droplet apparatus",
         hardware="MDCA in CIR", fuels="Burning liquid fuel droplets",
         conditions="Low-temperature (cool flame) combustion regime", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Droplet burning behavior",
         objective="Study cool-flame burning of liquid fuel droplets in microgravity.",
         variables="Droplet and atmosphere conditions.",
         earth="Engine and fuel combustion models", earth_cited=False,
         source=_BRIEF, s=(2, 0, 1, 2, 4, 2, 0)),
    dict(id="BRE", name="ACME · BRE", full="Burning Rate Emulator", status="archived", fuel="Gas",
         pillar="Prevention", facility_group="CIR · ACME gaseous-flame hardware",
         hardware="ACME modular hardware in CIR", fuels="Gaseous fuel emulating a burning material",
         conditions="Still atmospheres; material properties as the variable", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Burning rate as a function of material properties",
         objective="Study materials flammability in still atmospheres as a function of material properties (gaseous fuel).",
         variables="Material properties (emulated).",
         earth="Material flammability screening", earth_cited=False,
         source=_BRIEF, s=(4, 0, 0, 2, 4, 4, 1)),
    dict(id="CFI-G", name="ACME · CFI-G", full="Cool Flames Investigation with Gases", status="archived", fuel="Gas",
         pillar="Prevention", facility_group="CIR · ACME gaseous-flame hardware",
         hardware="ACME spherical burner in CIR", fuels="Flammable gas (e.g. propane) in nitrogen; inside-out flame",
         conditions="O₂-N₂ mixture of up to 85% O₂ flowing from the spherical burner", o2_max=85.0,
         gravity_regime="micro", n_exp=1,
         measurements="Flame structure at very high oxygen",
         objective="Highest-oxygen microgravity tests in PSI, using an inside-out flame from a spherical burner.",
         variables="O₂ concentration up to 85%.",
         earth="High-oxygen industrial and medical-gas safety", earth_cited=False,
         source=_BRIEF, s=(3, 0, 1, 5, 4, 2, 1)),
    dict(id="E-FIELD", name="ACME · E-FIELD", full="Electric-Field Effects on Laminar Diffusion Flames", status="archived",
         fuel="Gas", pillar="Prevention", facility_group="CIR · ACME gaseous-flame hardware",
         hardware="ACME modular hardware in CIR", fuels="Gaseous laminar diffusion flames",
         conditions="Electric field acting on flame ions mimics partial gravity", o2_max=np.nan,
         gravity_regime="analog", n_exp=1,
         measurements="Flame behavior vs electric-field strength",
         objective="Mimic partial gravity through the electric-field effect on naturally occurring flame ions.",
         variables="Electric-field strength.",
         earth="Flame control with electric fields", earth_cited=False,
         source=_BRIEF, s=(2, 0, 0, 1, 3, 1, 4)),
    dict(id="ACME-oth", name="ACME · other 3", full="Remaining ACME experiments", status="archived", fuel="Gas",
         pillar="Prevention", facility_group="CIR · ACME gaseous-flame hardware",
         hardware="ACME modular hardware in CIR", fuels="Gaseous fuels", conditions="See PSI per experiment",
         o2_max=np.nan, gravity_regime="micro", n_exp=3,
         measurements="Flame structure and stability (see PSI)",
         objective="Three further ACME experiments on gaseous flames; ACME aims at efficient, low-emission terrestrial combustion.",
         variables="Per experiment (see PSI).",
         earth="Cleaner and more efficient combustion", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(1, 0, 0, 1, 3, 1, 0)),
    dict(id="SPICE", name="SPICE", full="Smoke Point In Co-flow Experiment", status="archived", fuel="Gas",
         pillar="Detection", facility_group="MSG · SPICE flow duct",
         hardware="MSG with the SPICE fan-driven flow duct", fuels="Gaseous fuel in co-flow",
         conditions="Varied flow; smoke-point transition", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Soot / smoke point; stability and extinction of jet diffusion flames",
         objective="Study the mechanisms controlling stability and extinction of jet diffusion flames (continues LSP on STS-94).",
         variables="Co-flow velocity and fuel conditions.",
         earth="Soot reduction and cleaner combustion", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(2, 2, 0, 0, 3, 2, 0)),
    dict(id="SLICE", name="SLICE", full="Structure & Liftoff In Combustion Experiment", status="archived", fuel="Gas",
         pillar="Prevention", facility_group="MSG · SPICE flow duct",
         hardware="MSG with the SPICE fan-driven flow duct", fuels="Gaseous fuel jets in co-flow",
         conditions="Coflow laminar diffusion flames; flame liftoff", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Flame structure and liftoff",
         objective="Coflow laminar diffusion flames are well suited to accurate numerical modeling.",
         variables="Co-flow conditions.",
         earth="Burner and combustor design", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(1, 0, 0, 0, 3, 1, 0)),
    dict(id="SoFIE", name="SoFIE", full="Solid Fuel Ignition and Extinction", status="current", fuel="Solid",
         pillar="Prevention", facility_group="ISS · SoFIE",
         hardware="ISS hardware (see NASA SoFIE page)", fuels="Solid fuels",
         conditions="Ignition and extinction behavior of solids in low gravity", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Fire growth, ignition and extinction",
         objective="Test whether the 1-g flame-propagation test in NASA-STD-6001 is conservative enough for low gravity.",
         variables="Solid fuel and atmosphere conditions.",
         earth="Material flammability standards", earth_cited=False,
         source=_PAGE, s=(5, 0, 2, 3, 4, 5, 1)),
    dict(id="FLARE", name="FLARE", full="Flammability Limits At Reduced-g Experiment", status="current", fuel="Solid",
         pillar="Prevention", facility_group="ISS · FLARE (international team)",
         hardware="International project (JAXA, NASA, ESA, CNES, DLR and universities)", fuels="Solid materials",
         conditions="Reduced gravity", o2_max=np.nan, gravity_regime="micro", n_exp=1,
         measurements="Material flammability limits in normal gravity and microgravity",
         objective="Correlate flammability limits in normal and micro gravity so microgravity limits can be estimated from ground data.",
         variables="Gravity level; material.",
         earth="Cheaper ground-based flammability screening", earth_cited=False,
         source=_PAGE, s=(4, 0, 0, 2, 5, 4, 3)),
    dict(id="FM2", name="FM2", full="Flammability of Materials on the Moon", status="planned", fuel="Solid",
         pillar="Prevention", facility_group="Lunar surface",
         hardware="Lunar surface experiment", fuels="Burning materials (e.g. SIBAL cotton-fiberglass fabric on the ground)",
         conditions="Lunar gravity (1/6 g)", o2_max=np.nan, gravity_regime="partial", n_exp=1,
         measurements="Material burning at lunar gravity",
         objective="First-ever combustion experiment on another planetary body, at 1/6 g.",
         variables="Gravity level (Moon vs microgravity vs 1 g).",
         earth="Bounds flammability between 0 g and 1 g", earth_cited=False,
         source=_BRIEF + "; " + _PAGE, s=(5, 0, 0, 3, 5, 5, 5)),
    dict(id="LUCI", name="LUCI", full="Lunar-g Combustion Investigation", status="planned", fuel="Solid",
         pillar="Prevention", facility_group="Lunar surface",
         hardware="To be defined (data expected in PSI)", fuels="To be confirmed",
         conditions="Lunar gravity", o2_max=np.nan, gravity_regime="partial", n_exp=1,
         measurements="To be confirmed",
         objective="Expected source of partial-gravity combustion data for PSI.",
         variables="Gravity level.",
         earth="Fills the partial-gravity data gap", earth_cited=False,
         source=_BRIEF, s=(4, 0, 0, 2, 4, 4, 5)),
]

REQUIRED_COLS = ["id", "name", "full", "status", "fuel", "pillar", "facility_group", "hardware", "fuels",
                 "conditions", "measurements", "objective", "earth", "variables", "gravity_regime", "n_exp"] + CRIT_KEYS


def default_df() -> pd.DataFrame:
    df = pd.DataFrame(RECORDS)
    sc = pd.DataFrame(df.pop("s").tolist(), columns=CRIT_KEYS)
    df = pd.concat([df, sc], axis=1)
    df["o2_max"] = df["o2_max"].astype(float)
    return df


def validate_user_df(raw: pd.DataFrame):
    """Validate an uploaded CSV. Returns (df, errors)."""
    errors = []
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        return None, ["Missing columns: " + ", ".join(missing)]
    df = raw.copy()
    for c in CRIT_KEYS + ["n_exp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            errors.append("Column '" + c + "' must be numeric.")
    for c in CRIT_KEYS:
        df[c] = df[c].clip(0, 5)
    bad = sorted(set(df["status"]) - set(STATUS))
    if bad:
        errors.append("Unknown status values: " + ", ".join(map(str, bad)) + " (use archived/current/planned)")
    badg = sorted(set(df["gravity_regime"]) - set(GRAVITY_LABEL))
    if badg:
        errors.append("Unknown gravity_regime values: " + ", ".join(map(str, badg)) + " (use micro/partial/analog)")
    if "o2_max" not in df.columns:
        df["o2_max"] = np.nan
    df["o2_max"] = pd.to_numeric(df["o2_max"], errors="coerce")
    if "earth_cited" not in df.columns:
        df["earth_cited"] = False
    if "source" not in df.columns:
        df["source"] = "User CSV"
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("")
    return (None if errors else df), errors


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_frame(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    w = np.array([float(weights.get(k, 0)) for k in CRIT_KEYS])
    S = df[CRIT_KEYS].to_numpy(float)
    tw = w.sum()
    score = (S @ w) / (5.0 * tw) * 100.0 if tw > 0 else np.zeros(len(df))
    contrib = S * w
    out = df.copy()
    out["score"] = score
    drivers = []
    for i in range(len(out)):
        idx = np.argsort(-contrib[i], kind="mergesort")[:2]
        drivers.append(", ".join(CRITERIA[CRIT_KEYS[j]][0] for j in idx if contrib[i][j] > 0))
    out["drivers"] = drivers
    out = out.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def contributions(row: pd.Series, weights: dict) -> pd.DataFrame:
    """Points each criterion contributes to the 0-100 score of one investigation."""
    tw = sum(float(weights.get(k, 0)) for k in CRIT_KEYS)
    rows = []
    for k in CRIT_KEYS:
        pts = (float(weights.get(k, 0)) * float(row[k]) / (5.0 * tw) * 100.0) if tw > 0 else 0.0
        rows.append({"criterion": CRITERIA[k][0], "points": pts})
    return pd.DataFrame(rows)


def similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity of the 7-criteria score vectors."""
    M = df[CRIT_KEYS].to_numpy(float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = M / norms
    return pd.DataFrame(U @ U.T, index=df["name"].tolist(), columns=df["name"].tolist())


def compare_text(sel: pd.DataFrame) -> list:
    """Plain-language shared / different facts for a selection of investigations."""
    out = []
    names = ", ".join(sel["name"])
    for col, label in [("facility_group", "hardware platform"), ("fuel", "fuel type"), ("pillar", "safety pillar"),
                       ("gravity_regime", "gravity regime")]:
        vals = sel[col].unique()
        if len(vals) == 1:
            shown = GRAVITY_LABEL.get(vals[0], vals[0]) if col == "gravity_regime" else vals[0]
            out.append("All of " + names + " share the same " + label + ": **" + str(shown) + "**, which makes their data directly comparable.")
        else:
            parts = []
            for v in vals:
                members = ", ".join(sel.loc[sel[col] == v, "name"])
                shown = GRAVITY_LABEL.get(v, v) if col == "gravity_regime" else v
                parts.append(str(shown) + " (" + members + ")")
            out.append("They differ in " + label + ": " + "; ".join(parts) + ".")
    return out


# --------------------------------------------------------------------------------------
# Habitats, evidence readiness and gaps
# --------------------------------------------------------------------------------------
HABITATS = {
    "ISS-like cabin": {"o2": 21.0, "psi": 14.7, "g": 0.0,
                       "note": "Earth-like air at sea-level pressure, microgravity."},
    "Moon base / pressurized rover": {"o2": 30.0, "psi": 10.2, "g": 0.16,
                                      "note": "Brief: 30% O₂ or more at 10.2 psi or less. Lunar gravity is 1/6 g."},
    "Mars surface habitat": {"o2": 30.0, "psi": 10.2, "g": 0.38,
                             "note": "Assumption: Moon-like atmosphere strategy (not specified in the brief). Mars gravity about 0.38 g."},
    "Spacesuit (lunar EVA)": {"o2": 100.0, "psi": 4.3, "g": 0.16,
                              "note": "Brief: 100% O₂ at 4.3 psi."},
}
PILLARS = ["prevention", "detection", "suppression"]


def habitat_table() -> pd.DataFrame:
    rows = []
    for name, h in HABITATS.items():
        rows.append({"Habitat": name, "O₂ (%)": h["o2"], "Pressure (psi)": h["psi"],
                     "O₂ partial pressure (psi)": round(h["o2"] / 100.0 * h["psi"], 2),
                     "O₂ vs Earth air": "{:+.0f}%".format((h["o2"] / 21.0 - 1) * 100),
                     "Gravity (g)": h["g"], "Note": h["note"]})
    return pd.DataFrame(rows)


def evidence_readiness(df: pd.DataFrame, hab: dict, pen_g: float = 0.5, pen_o2: float = 0.4) -> dict:
    """Evidence Readiness Index (0-100) per pillar: how well the archived PSI evidence supports a habitat.

    base  = mean of the 3 best archived investigations for the pillar (0-5 -> 0-100)
    gravity penalty  applies when the habitat has gravity > 0 and PSI has no direct partial-g data
    oxygen penalty   applies when the habitat is oxygen-enriched; reduced if a documented test reaches that O2
    """
    arch = df[df["status"] == "archived"]
    n_partial = int((arch["gravity_regime"] == "partial").sum())
    n_analog = int((arch["gravity_regime"] == "analog").sum())
    if hab["g"] == 0:
        gs = 0.0
    elif n_partial > 0:
        gs = 0.4
    elif n_analog > 0:
        gs = 0.8
    else:
        gs = 1.0
    o2 = hab["o2"]
    os_ = 0.0 if o2 <= 21.5 else (0.6 if o2 <= 40 else 1.0)
    doc_max = arch["o2_max"].max()
    if os_ > 0 and pd.notna(doc_max) and doc_max >= o2:
        os_ *= 0.7
    out = {}
    for p in PILLARS:
        top = arch[p].nlargest(3).mean() if len(arch) else 0.0
        base = float(top) / 5.0 * 100.0
        out[p] = {"value": base * (1 - pen_g * gs) * (1 - pen_o2 * os_), "base": base,
                  "gravity_severity": gs, "oxygen_severity": os_}
    return out


def gap_report(df: pd.DataFrame) -> list:
    """Data-driven list of evidence gaps. Each item: severity (critical/high/medium), title, detail."""
    gaps = []
    arch = df[df["status"] == "archived"]
    planned = df[df["status"] == "planned"]
    n_partial = int((arch["gravity_regime"] == "partial").sum())
    if n_partial == 0:
        names = ", ".join(planned.loc[planned["gravity_regime"] == "partial", "name"]) or "none listed"
        analog = ", ".join(arch.loc[arch["gravity_regime"] == "analog", "name"]) or "none"
        gaps.append({"severity": "critical", "title": "No partial-gravity combustion data in PSI",
                     "detail": "Lunar (1/6 g) and Martian gravity are unrepresented. Expected: " + names +
                               ". Closest analog today: " + analog + ". Microgravity and 1 g results can only bound the intermediate range."})
    solid_hi = arch[(arch["fuel"] == "Solid") & (arch["o2_max"] >= 30)]
    hi = arch[arch["o2_max"] >= 30]
    if len(solid_hi) == 0:
        who = ", ".join(hi["name"]) or "none"
        gaps.append({"severity": "high", "title": "No documented solid-fuel test at 30% O₂ or more",
                     "detail": "Documented high-O₂ evidence: " + who + " (gaseous fuel, up to 85% O₂). Moon habitats are expected at 30% or more, "
                               "so run-level O₂ of the solid-fuel tests (BASS, Saffire) must be checked in the PSI datasets."})
    sol_supp = arch[(arch["suppression"] >= 4) & (arch["fuel"] == "Solid")]
    if len(sol_supp) == 0:
        liq = ", ".join(arch.loc[arch["suppression"] >= 4, "name"]) or "none"
        gaps.append({"severity": "medium", "title": "Suppressant effectiveness is documented on droplets, not solids",
                     "detail": "Strong suppression evidence comes from " + liq + " (burning liquid droplets with CO₂ / helium dilution). "
                               "Solid-material extinction data exist but are less direct."})
    by_fac = arch.groupby("facility_group")["n_exp"].sum().sort_values(ascending=False)
    if len(by_fac) >= 2:
        top3 = by_fac.head(3)
        share = float(top3.sum()) / float(by_fac.sum()) * 100.0
        gaps.append({"severity": "medium", "title": "Evidence concentrated on a few hardware platforms",
                     "detail": "{:.0f}% of archived experiments ran on three platforms: ".format(share) +
                               "; ".join(str(k) + " (" + str(int(v)) + ")" for k, v in top3.items()) +
                               ". Shared hardware makes runs comparable, but also shares systematic limits."})
    return gaps


# --------------------------------------------------------------------------------------
# Resources and Earth impact
# --------------------------------------------------------------------------------------
RESOURCES = [
    ("PSI Data Repository: Combustion Science", "https://psi.nasa.gov/physci/repo/search?q=combustion", "Source of the data."),
    ("NASA Combustion Science", "https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/combustion-science/", "Overview of most investigations."),
    ("ScienceCasts: The Power of Studying Combustion on the ISS", "https://www.youtube.com/watch?v=W1oXykoz9zg", "3.6-minute video (2018)."),
    ("A Researcher's Guide to ISS Combustion Science", "https://www.nasa.gov/missions/station/a-researchers-guide-to-combustion-science/", "2015 review, 50+ pages."),
    ("Apollo 1", "https://sma.nasa.gov/sma-disciplines/accllp/apollo-1", "Pre-launch fire, lessons learned."),
    ("Apollo 13", "https://sma.nasa.gov/sma-disciplines/accllp/apollo-13", "Oxygen tank explosion, lessons learned."),
    ("25 Years Ago: Fire Aboard Space Station Mir", "https://www.nasa.gov/history/25-years-ago-fire-aboard-space-station-mir/", "Oxygen-generator fire, 1997."),
    ("Fireman's Air Tanks", "https://spinoff.nasa.gov/node/9582", "1970s spacesuit tech for firefighters' breathing gear."),
    ("Fire Detector", "https://spinoff.nasa.gov/node/9407", "1970s shuttle smoke detector; discriminates smoke from dust."),
    ("New Technology for the Fire Attire", "https://spinoff.nasa.gov/node/9918", "1980s spacesuit tech for firefighter clothing."),
]

# year used for plotting; "approx" marks decade-level dates given in the sources
EVENTS = [
    {"year": 1967, "kind": "Incident", "label": "Apollo 1", "approx": False,
     "detail": "Fire erupted in the oxygen-filled spacecraft during a pre-launch test; the crew died."},
    {"year": 1970, "kind": "Incident", "label": "Apollo 13", "approx": False,
     "detail": "Oxygen tank explosion on the way to the Moon; the crew survived."},
    {"year": 1975, "kind": "Spin-off", "label": "Firefighter air tanks", "approx": True,
     "detail": "1970s adaptation of spacesuit technology to firefighters' air tanks and breathing gear."},
    {"year": 1977, "kind": "Spin-off", "label": "Smoke / dust-discriminating detector", "approx": True,
     "detail": "1970s shuttle smoke detector; fewer nuisance alarms, with industrial uses such as mining."},
    {"year": 1985, "kind": "Spin-off", "label": "Firefighter protective clothing", "approx": True,
     "detail": "1980s application of spacesuit technology to firefighter clothing."},
    {"year": 1997, "kind": "Incident", "label": "Mir fire", "approx": False,
     "detail": "Oxygen-generator failure caused a fire; the multinational crew survived."},
    {"year": 2026, "kind": "Incident", "label": "Artemis II smoke alarm", "approx": False,
     "detail": "A smoke alarm went off about 80,000 miles from home, per the mission commander."},
]

# --------------------------------------------------------------------------------------
# Local "SI" engine (retrieval + rules). Used when no LLM key is configured.
# --------------------------------------------------------------------------------------
STOP = {"the", "and", "for", "what", "which", "that", "this", "with", "are", "about", "tell", "give", "show",
        "please", "any", "how", "does", "from", "into", "have", "has", "was", "were", "can", "you", "data"}


def _line(r) -> str:
    return "- **{}** [{}] ({}, {}): {}".format(r["name"], r["id"], r["pillar"], STATUS[r["status"]]["label"], r["objective"])


def search(df: pd.DataFrame, q: str, k: int = 4) -> pd.DataFrame:
    toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2 and t not in STOP]
    if not toks:
        return df.head(0)
    cols = ["id", "name", "full", "fuels", "conditions", "measurements", "objective", "hardware", "facility_group", "variables"]
    text = df[cols].astype(str).agg(" ".join, axis=1).str.lower()
    hits = text.apply(lambda t: sum(t.count(tok) for tok in toks))
    out = df.assign(_hit=hits)
    out = out[out["_hit"] > 0].sort_values("_hit", ascending=False, kind="mergesort")
    return out.head(k)


def _match_ids(df: pd.DataFrame, q: str) -> list:
    special = {"Saffire": r"saffire", "DAFT": r"daft", "ACME-oth": r"acme-oth", "E-FIELD": r"e-?field"}
    found = []
    for _, r in df.iterrows():
        rid = str(r["id"])
        if rid in special:
            pat, flags = special[rid], re.I
        else:
            pat = r"(?<![\w-])" + re.escape(rid) + r"(?![\w-])"
            flags = 0 if rid == "SAME" else re.I
        if re.search(pat, q, flags):
            found.append(rid)
    return found


def answer_local(q: str, df: pd.DataFrame, weights: dict) -> str:
    ql = q.lower()
    sc = score_frame(df, weights)

    ids = _match_ids(df, q)
    if ids:
        parts = []
        for rid in ids[:3]:
            r = df[df["id"] == rid].iloc[0]
            parts.append("**{} [{}]** — {}\n\n- Status: {}\n- Hardware: {}\n- Fuels: {}\n- Conditions: {}\n- Measurements: {}\n- Varied: {}\n- Earth link: {}".format(
                r["full"], r["id"], r["objective"], STATUS[r["status"]]["label"], r["hardware"], r["fuels"],
                r["conditions"], r["measurements"], r["variables"], r["earth"]))
        return "\n\n---\n\n".join(parts)

    if "acme" in ql:
        rows = df[df["facility_group"].str.contains("ACME")]
        return "All ACME experiments ran on the same modular hardware in the Combustion Integrated Rack with gaseous fuels:\n" + "\n".join(_line(r) for _, r in rows.iterrows())

    if re.search(r"gap|missing|lack|weak|limitation|unknown", ql):
        g = gap_report(df)
        return "Main evidence gaps found in the dataset:\n" + "\n".join("- **{}** ({}): {}".format(x["title"], x["severity"], x["detail"]) for x in g)

    if re.search(r"hardware|facility|rack|glovebox|same apparatus|comparable", ql):
        grp = df.groupby("facility_group")["name"].apply(lambda s: ", ".join(s))
        return "Investigations grouped by shared hardware (same hardware means directly comparable runs):\n" + "\n".join("- **{}**: {}".format(k, v) for k, v in grp.items())

    if re.search(r"partial|lunar|mars|gravity|luci|fm2|1/6", ql):
        part = df[df["gravity_regime"].isin(["partial", "analog"])]
        return ("PSI currently holds **no** partial-gravity combustion data. Expected or analog sources:\n" +
                "\n".join(_line(r) for _, r in part.iterrows()) +
                "\n\nMicrogravity and normal-gravity results can bound what happens at intermediate gravity.")

    if re.search(r"oxygen|o2|moon|rover|habitat|30%|enrich", ql):
        t = sc.sort_values(["highO2", "score"], ascending=False, kind="mergesort").head(3)
        h = HABITATS["Moon base / pressurized rover"]
        return ("For oxygen-enriched habitats the most relevant investigations are:\n" + "\n".join(_line(r) for _, r in t.iterrows()) +
                "\n\nA Moon base is expected at 30% O₂ or more and 10.2 psi or less: about {:+.0f}% O₂ versus Earth air "
                "(O₂ partial pressure {:.2f} psi vs {:.2f} psi). Only CFI-G documents up to 85% O₂, with a gaseous fuel.".format(
                    (h["o2"] / 21 - 1) * 100, h["o2"] / 100 * h["psi"], 0.21 * 14.7))

    if re.search(r"smoke|detect|alarm|sensor", ql):
        t = sc.sort_values(["detection", "score"], ascending=False, kind="mergesort").head(3)
        return "Best evidence for detection:\n" + "\n".join(_line(r) for _, r in t.iterrows())

    if re.search(r"dust", ql):
        r = df[df["id"] == "DAFT"].iloc[0]
        return "Dust detection:\n" + _line(r) + "\n\nThe SAME smoke instrument was reused to detect dust."

    if re.search(r"suppress|extinguish|inert|co2|helium|put out", ql):
        t = sc.sort_values(["suppression", "score"], ascending=False, kind="mergesort").head(3)
        return "Best evidence for suppression:\n" + "\n".join(_line(r) for _, r in t.iterrows())

    if re.search(r"model|predict|simulat|validate", ql):
        t = sc.sort_values(["modeling", "score"], ascending=False, kind="mergesort").head(4)
        return "Best datasets for predictive modeling:\n" + "\n".join(_line(r) for _, r in t.iterrows())

    if re.search(r"earth|terrestrial|spin|everyday|firefight", ql):
        cited = df[df["earth_cited"].astype(bool)]
        extra = "\n".join("- **{}**: {}".format(r["name"], r["earth"]) for _, r in cited.iterrows())
        return ("NASA fire-safety work has already benefited Earth: fine-water-mist extinguishers, better smoke detection, and firefighter gear "
                "(protective clothing and air tanks).\n" + extra)

    if re.search(r"best|top|rank|recommend|priorit", ql):
        t = sc.head(3)
        return "Top investigations for the current profile:\n" + "\n".join("- **{}** ({:.0f}/100): {}".format(r["name"], r["score"], r["drivers"]) for _, r in t.iterrows())

    hits = search(df, q)
    if len(hits):
        return "Closest matches in the dataset:\n" + "\n".join(_line(r) for _, r in hits.iterrows())
    r0 = sc.iloc[0]
    return ("I could not match that question to the dataset. With the current profile the top investigation is **{}**. "
            "Try asking about oxygen, smoke detection, suppression, partial gravity, dust, modeling, hardware or gaps.".format(r0["name"]))


# --------------------------------------------------------------------------------------
# LLM (optional): grounded in the dataset
# --------------------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are the IGNIS NOVA SI Analyst for the NASA Space Apps challenge 'Flame in Freefall'. "
    "Answer questions about combustion investigations for spacecraft fire safety using ONLY the dataset provided. "
    "Cite investigations by their id in square brackets, e.g. [FLEX]. If the dataset does not contain the answer, say so plainly. "
    "Never invent measurements, dates or results. Remember: PSI currently has no partial-gravity combustion data. "
    "Scores are analyst-assigned relevance ratings from 0 to 5, not measured quantities. Be concise and practical for engineers."
)


def llm_context(df: pd.DataFrame, weights: dict, profile_name: str) -> str:
    lines = ["Mission profile: " + profile_name, "Weights: " + ", ".join(k + "=" + str(weights[k]) for k in CRIT_KEYS), "", "DATASET:"]
    for _, r in df.iterrows():
        sc = ", ".join(k + "=" + str(int(r[k])) for k in CRIT_KEYS)
        lines.append("[{}] {} | status={} | fuel={} | pillar={} | hardware={} | fuels={} | conditions={} | measures={} | objective={} | scores: {}".format(
            r["id"], r["full"], r["status"], r["fuel"], r["pillar"], r["hardware"], r["fuels"], r["conditions"],
            r["measurements"], r["objective"], sc))
    lines.append("")
    lines.append("HABITATS: " + "; ".join("{}: {}% O2, {} psi, {} g".format(k, v["o2"], v["psi"], v["g"]) for k, v in HABITATS.items()))
    return "\n".join(lines)


def ask_llm(question: str, df: pd.DataFrame, weights: dict, profile_name: str, api_key: str,
            model: str = "claude-sonnet-5-5") -> str:
    import anthropic  # imported lazily so the app still runs without the package

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=900, system=LLM_SYSTEM,
        messages=[{"role": "user", "content": llm_context(df, weights, profile_name) + "\n\nQUESTION: " + question}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


# --------------------------------------------------------------------------------------
# Exportable brief
# --------------------------------------------------------------------------------------
def make_brief(scored: pd.DataFrame, profile_name: str, weights: dict, df_all: pd.DataFrame, top_n: int = 5) -> str:
    L = ["# IGNIS NOVA — Fire Safety Evidence Brief", "", "**Profile:** " + profile_name, ""]
    L.append("**Weights:** " + ", ".join(CRITERIA[k][0] + " " + str(weights[k]) for k in CRIT_KEYS))
    L += ["", "## Top investigations", ""]
    for _, r in scored.head(top_n).iterrows():
        L.append("{}. **{}** ({}) — {:.0f}/100. {} Drivers: {}.".format(r["rank"], r["name"], r["full"], r["score"], r["objective"], r["drivers"] or "n/a"))
    L += ["", "## Evidence readiness (0-100, default penalties)", ""]
    for name, h in HABITATS.items():
        er = evidence_readiness(df_all, h)
        L.append("- **{}**: ".format(name) + ", ".join("{} {:.0f}".format(p, er[p]["value"]) for p in PILLARS))
    L += ["", "## Evidence gaps", ""]
    for g in gap_report(df_all):
        L.append("- **{}** ({}): {}".format(g["title"], g["severity"], g["detail"]))
    L += ["", "_Scores are analyst-assigned 0-5 relevance ratings, not measurements. Verify details in the PSI repository._"]
    return "\n".join(L)
