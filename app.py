import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import plotly.graph_objects as go
import itertools

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("_page_cfg_done", False):
    try:
        st.set_page_config(page_title="BIO4EEB Performance Tool", layout="wide")
    except Exception:
        pass
    st.session_state["_page_cfg_done"] = True

# ────────────────────────────────────────────────────────────────────────────
# FILES & CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
# Map KPI -> trained artifact path (we resolve across folders automatically)
MODEL_MAP = {
    "EUI": "models/ann_model_bundle_output_EUI.pkl",
    "CD":  "models/rf_model_output_eui_cooling_floor_0.pkl",
    "HD":  "models/ann_model_bundle_output_eui_heating_floor_0.pkl",
    "PMV": "models/ann_model_bundle_output_pmv.pkl",
    "PPD": "models/ann_model_bundle_output_ppd.pkl",
}
SCALER_PATH = "standard_scaler.pkl"

BASELINES = {
    "EUI": 154.42,   # kWh/m²·yr
    "CD":  14.40,    # kWh/m²
    "HD":  85.45,    # kWh/m²
    "PMV": 47.21,    # %
    "PPD": 46.03,    # %
}

# Folders to search for artifacts (per your notebook saving)
ARTIFACT_DIRS = ["models", "saved_models", "saved_ann_models", "."]

# ────────────────────────────────────────────────────────────────────────────
# LOOK-UP TABLES
# ────────────────────────────────────────────────────────────────────────────
# IMPORTANT: tokens here are UI keys. We map them to training tokens where needed.
INSULATION_PROPERTIES = {
    "No_insulation": {"thermal_conductivity": 0, "density": 0, "specific_heat": 0},
    "EPS": {"thermal_conductivity": 0.035, "density": 25, "specific_heat": 1400},
    "XPS": {"thermal_conductivity": 0.033, "density": 35, "specific_heat": 1300},
    "Glass wool": {"thermal_conductivity": 0.04, "density": 15, "specific_heat": 840},
    "BioPUR": {"thermal_conductivity": 0.022, "density": 45, "specific_heat": 1600},
    "PLA foam": {"thermal_conductivity": 0.032, "density": 65, "specific_heat": 1400},
    "Posidonia Panel": {"thermal_conductivity": 0.044, "density": 100, "specific_heat": 1935},

    # Facades (UI keys) — numeric props from notebook; thickness handled separately
    "Facade_Type 1": {"thermal_conductivity": 0.023, "density": 236.72, "specific_heat": 832},
    "Facade_Type 2": {"thermal_conductivity": 0.025, "density": 210.94, "specific_heat": 848},
    "Facade_Type 3": {"thermal_conductivity": 0.061, "density": 231.00, "specific_heat": 948},
    "Facade_Type 4": {"thermal_conductivity": 0.182, "density": 245.00, "specific_heat": 926},
    "Facade_Type 5": {"thermal_conductivity": 0.199, "density": 231.41, "specific_heat": 947},
}

# Facade trained thickness (from WALL_MAP in the notebook)
FACADE_TRAINED_THICKNESS = {
    "Facade_Type 1": 0.1295,
    "Facade_Type 2": 0.1495,
    "Facade_Type 3": 0.1695,
    "Facade_Type 4": 0.1660,
    "Facade_Type 5": 0.1860,
}

FACADE_TYPES = list(FACADE_TRAINED_THICKNESS.keys())

WINDOW_PROPERTIES = {
    "Window_Existing": {"U_value": 3.0, "SHGC": 0.64, "VT": 0.65},
    "Window_BIO4EEB W2": {"U_value": 1.1, "SHGC": 0.6, "VT": 0.81},
    "Window_BIO4EEB W1": {"U_value": 1.1, "SHGC": 0.4, "VT": 0.69},
    "Window_Retrofit 1": {"U_value": 1.6, "SHGC": 0.4, "VT": 0.55},
    "Window_Retrofit 2": {"U_value": 1.0, "SHGC": 0.38, "VT": 0.47},
}

# ────────────────────────────────────────────────────────────────────────────
# LOAD ASSETS
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Failed to load {path}: {exc}")
        return None

def resolve_artifact_path(fname: str) -> Optional[str]:
    # If user passed an absolute or already valid relative path
    if os.path.exists(fname):
        return fname
    # Search in known folders
    for d in ARTIFACT_DIRS:
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            return candidate
    return None

def unwrap_model(obj):
    """
    Supports both:
    - direct estimator
    - bundle dict: {'model': estimator, 'scaler': scaler}
    """
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("scaler", None)
    return obj, None

@st.cache_resource(show_spinner=True)
def load_models(model_map: dict):
    mdls = {}
    for k, fname in model_map.items():
        p = resolve_artifact_path(fname)
        if not p:
            st.error(f"Missing model file: {fname} (searched: {ARTIFACT_DIRS})")
            st.stop()
        obj = load_pickle(p)
        if obj is None:
            st.stop()
        mdl, _maybe_scaler = unwrap_model(obj)
        mdls[k] = mdl
    return mdls

# Load scaler (handle bundle-style if someone saved it that way)
scaler_path = resolve_artifact_path(SCALER_PATH)
if not scaler_path:
    st.error(f"Missing scaler file: {SCALER_PATH} (searched: {ARTIFACT_DIRS})")
    st.stop()

scaler_obj = load_pickle(scaler_path)
if scaler_obj is None:
    st.stop()

# If someone saved a bundle as scaler pickle (rare), unwrap it
if isinstance(scaler_obj, dict) and "scaler" in scaler_obj:
    scaler = scaler_obj["scaler"]
else:
    scaler = scaler_obj

EXPECTED_COLS = list(getattr(scaler, "feature_names_in_", []))
if not EXPECTED_COLS:
    st.error("Scaler lacks feature_names_in_. Re-train scaler or hard-code EXPECTED_COLS.")
    st.stop()

models = load_models(MODEL_MAP)

# ────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# ────────────────────────────────────────────────────────────────────────────
def is_baseline_case(wall_type: str, roof_type: str, window_type: str) -> bool:
    return (wall_type == "No_insulation") and (roof_type == "No_insulation") and (window_type == "Window_Existing")

def to_percent_safe(raw: float) -> float:
    """Robust conversion to 0–100%. Handles prob, %, and unbounded scores."""
    if 0.0 <= raw <= 1.0:
        prob = raw
    elif 0.0 <= raw <= 100.0:
        return float(np.clip(raw, 0.0, 100.0))
    else:
        prob = 1.0 / (1.0 + np.exp(-raw))
    return float(np.clip(prob * 100.0, 0.0, 100.0))

def norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def set_first_existing(row: dict, candidates: list[str], value: float) -> Optional[str]:
    for c in candidates:
        if c in row:
            row[c] = value
            return c
    return None

def find_by_normalized(row: dict, prefix: str, target_label: str) -> Optional[str]:
    """
    Find a column by normalized matching, within a prefix family.
    Example: find_by_normalized(row, 'input_window_', 'Window_BIO4EEB W1')
    """
    tgt = norm_key(target_label)
    best = None
    for c in row.keys():
        if not c.startswith(prefix):
            continue
        if tgt in norm_key(c):
            best = c
            break
    return best

# Training token mapping (from notebook)
INS_TOKEN_MAP = {
    "No_insulation": "No insulation",
    "EPS": "EPS",
    "XPS": "XPS",
    "Glass wool": "Glass wool",
    "BioPUR": "BioPUR",
    "PLA foam": "PLAfoam",
    "Posidonia Panel": "PosidoniaPanel",
    "Facade_Type 1": "Facade_type 1",
    "Facade_Type 2": "Facade_type 2",
    "Facade_Type 3": "Facade_type 3",
    "Facade_Type 4": "Facade_type 4",
    "Facade_Type 5": "Facade_type 5",
}

def set_refurbished_components(row: dict, wall_type: str, roof_type: str, window_type: str) -> Optional[str]:
    """
    Notebook has columns like:
    Refurbished components_Existing
    Refurbished components_Only Wall / Only Roof / Only Window / Only Facade
    Refurbished components_Wall+Roof / Wall+Window / Roof+Window / Wall+Window+Roof
    Refurbished components_Facade+Roof / Facade+Window / Facade+Roof+Window
    """
    parts = []
    if wall_type != "No_insulation":
        parts.append("Facade" if wall_type in FACADE_TYPES else "Wall")
    if roof_type != "No_insulation":
        parts.append("Roof")
    if window_type != "Window_Existing":
        parts.append("Window")

    # If none selected
    if not parts:
        candidates = [
            "Refurbished components_Existing",
            "Refurbished components_existing",
        ]
        return set_first_existing(row, candidates, 1.0)

    # If single
    if len(parts) == 1:
        p = parts[0]
        candidates = [
            f"Refurbished components_Only {p}",
            f"Refurbished components_Only {p.lower()}",
            f"Refurbished components_Only_{p}",
            f"Refurbished components_Only_{p.lower()}",
        ]
        return set_first_existing(row, candidates, 1.0)

    # If 2 or 3: try permutations until a matching column exists
    best_col = None
    for perm in itertools.permutations(parts):
        key = "+".join(perm)
        candidates = [
            f"Refurbished components_{key}",
            f"Refurbished components_{key.replace(' ', '')}",
            f"Refurbished components_{key.replace(' ', '_')}",
        ]
        best_col = set_first_existing(row, candidates, 1.0)
        if best_col:
            break
    return best_col

def set_wall_roof_onehot(row: dict, kind: str, material: str) -> Optional[str]:
    token = INS_TOKEN_MAP.get(material, material)
    candidates = [
        f"{kind}_insulation_{token}",
        f"{kind}_insulation_{token.replace(' ', '_')}",
        f"{kind}_insulation_{token.replace('_', ' ')}",
    ]
    # If facade token: also try some alternative spellings (defensive)
    if "Facade" in token:
        num = "".join(ch for ch in token if ch.isdigit())
        candidates += [
            f"{kind}_insulation_Facade_type {num}",
            f"{kind}_insulation_Facade_type_{num}",
            f"{kind}_insulation_FacadeType{num}",
        ]
    for c in candidates:
        if c in row:
            row[c] = 1.0
            return c
    # fallback scan in schema by normalized matching
    prefix = f"{kind}_insulation_"
    hit = find_by_normalized(row, prefix, token)
    if hit:
        row[hit] = 1.0
        return hit
    return None

def set_window_onehot(row: dict, window_type: str) -> Optional[str]:
    # Most common: input_window_<name>
    hit = find_by_normalized(row, "input_window_", window_type)
    if hit:
        row[hit] = 1.0
        return hit
    # fallback: scan any column containing "window" and the label
    for c in row.keys():
        if "window" in c.lower() and norm_key(window_type) in norm_key(c):
            row[c] = 1.0
            return c
    return None

def build_feature_row(
    wall_type: str,
    wall_thickness: float,
    roof_type: str,
    roof_thickness: float,
    window_type: str,
    enable_proxy: bool = True,
) -> dict:
    row = {c: 0.0 for c in EXPECTED_COLS}
    missing_flags = []

    # Enforce: facades are wall-only
    if roof_type in FACADE_TYPES:
        roof_type = "No_insulation"
        roof_thickness = 0.0
        missing_flags.append("Roof facade detected → forced to No_insulation (facade is wall-only).")

    # Refurbished components_* one-hot (schema-aware)
    refurb_col = set_refurbished_components(row, wall_type, roof_type, window_type)

    # One-hot wall/roof insulation
    wall_col = set_wall_roof_onehot(row, "Wall", wall_type)
    roof_col = set_wall_roof_onehot(row, "Roof", roof_type)

    if wall_type != "No_insulation" and wall_col is None:
        # Optional proxy only for BioPUR if schema lacks explicit BioPUR column
        if enable_proxy and wall_type == "BioPUR":
            for tgt in ("Wall_insulation_XPS", "Wall_insulation_EPS"):
                if tgt in row:
                    row[tgt] = 1.0
                    missing_flags.append(f"Wall_insulation 'BioPUR' not in schema → proxied to '{tgt.split('_')[-1]}'.")
                    wall_col = tgt
                    break
        if wall_col is None:
            missing_flags.append(f"Wall one-hot not found for '{wall_type}' → numeric props only.")

    if roof_type != "No_insulation" and roof_col is None:
        if enable_proxy and roof_type == "BioPUR":
            for tgt in ("Roof_insulation_XPS", "Roof_insulation_EPS"):
                if tgt in row:
                    row[tgt] = 1.0
                    missing_flags.append(f"Roof_insulation 'BioPUR' not in schema → proxied to '{tgt.split('_')[-1]}'.")
                    roof_col = tgt
                    break
        if roof_col is None:
            missing_flags.append(f"Roof one-hot not found for '{roof_type}' → numeric props only.")

    # Window one-hot
    win_col = set_window_onehot(row, window_type)
    if window_type != "Window_Existing" and win_col is None:
        missing_flags.append(f"Window one-hot not found for '{window_type}'.")

    # Numeric features (set only if column exists)
    wp = INSULATION_PROPERTIES[wall_type]
    rp = INSULATION_PROPERTIES[roof_type]
    winp = WINDOW_PROPERTIES[window_type]

    # Wall thickness: if facade, override to trained thickness
    if wall_type in FACADE_TYPES:
        wall_thickness = float(FACADE_TRAINED_THICKNESS[wall_type])

    # Wall numeric
    set_first_existing(row, ["Wall_therm_cond"], wp["thermal_conductivity"])
    set_first_existing(row, ["Wall_thickness"], 0.0 if wall_type == "No_insulation" else float(wall_thickness))
    set_first_existing(row, ["Wall_density"], wp["density"])
    set_first_existing(row, ["Wall_heat_cap"], wp["specific_heat"])

    # Roof numeric
    set_first_existing(row, ["Roof_therm_cond"], rp["thermal_conductivity"])
    set_first_existing(row, ["Roof_thickness"], 0.0 if roof_type == "No_insulation" else float(roof_thickness))
    set_first_existing(row, ["Roof_density"], rp["density"])
    set_first_existing(row, ["Roof_heat_cap"], rp["specific_heat"])

    # Window numeric (some schemas may vary: try common variants)
    set_first_existing(row, ["Visible_transmittance", "Visible Transmittance", "Visible_transmittance "], winp["VT"])
    set_first_existing(row, ["Window_U_value", "Window_U-value", "Window_Uvalue"], winp["U_value"])

    # Avoid accidental outputs
    for c in EXPECTED_COLS:
        if c.startswith("output_") or c.startswith("out_"):
            row[c] = 0.0

    st.session_state["_missing_flags"] = missing_flags
    st.session_state["_refurb_col_used"] = refurb_col
    st.session_state["_onehot_wall_col_used"] = wall_col
    st.session_state["_onehot_roof_col_used"] = roof_col
    st.session_state["_onehot_window_col_used"] = win_col
    return row

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Material & Window Selection")

wall_type = st.sidebar.selectbox("Wall insulation type", list(INSULATION_PROPERTIES.keys()))

# Wall thickness control
if wall_type == "No_insulation":
    wall_thickness = 0.0
    st.sidebar.number_input("Wall insulation thickness (m)", value=0.0, step=0.01, format="%.4f", disabled=True)
elif wall_type in FACADE_TYPES:
    wall_thickness = float(FACADE_TRAINED_THICKNESS[wall_type])
    st.sidebar.number_input(
        "Wall thickness (m) — trained value",
        value=wall_thickness,
        step=0.0005,
        format="%.4f",
        disabled=True
    )
else:
    wall_thickness = st.sidebar.slider("Wall insulation thickness (m)", 0.01, 0.30, 0.05, 0.01)

# Roof options exclude facades
roof_options = [k for k in INSULATION_PROPERTIES.keys() if k not in set(FACADE_TYPES)]
roof_type = st.sidebar.selectbox("Roof insulation type", roof_options)

if roof_type == "No_insulation":
    roof_thickness = 0.0
    st.sidebar.number_input("Roof insulation thickness (m)", value=0.0, step=0.01, format="%.2f", disabled=True)
else:
    roof_thickness = st.sidebar.slider("Roof insulation thickness (m)", 0.01, 0.30, 0.05, 0.01)

window_type = st.sidebar.selectbox("Window type", list(WINDOW_PROPERTIES.keys()))

show_diag = st.sidebar.checkbox("Show diagnostics table (advanced)", value=False)

# ────────────────────────────────────────────────────────────────────────────
# BUILD FEATURES
# ────────────────────────────────────────────────────────────────────────────
raw_row = build_feature_row(wall_type, wall_thickness, roof_type, roof_thickness, window_type, enable_proxy=True)
raw_df = pd.DataFrame([raw_row], columns=EXPECTED_COLS)
scaled_df = pd.DataFrame(scaler.transform(raw_df), columns=EXPECTED_COLS)

# ────────────────────────────────────────────────────────────────────────────
# PREDICTIONS (single scenario)
# ────────────────────────────────────────────────────────────────────────────
preds = {}
_debug = {}

for metric, mdl in models.items():

    def _predict(df):
        if hasattr(mdl, "feature_names_in_"):
            Xloc = df[list(mdl.feature_names_in_)]
        else:
            Xloc = df.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return float(mdl.predict(Xloc)[0])

    if metric in {"EUI", "CD", "HD"}:
        y_model = _predict(scaled_df)
        preds[metric] = y_model
        _debug[metric] = {"source": "scaled", "model_output": y_model, "postproc": "identity", "display": y_model}
    else:
        score_raw = _predict(raw_df)
        score_scl = _predict(scaled_df)
        pct_raw = to_percent_safe(score_raw)
        pct_scl = to_percent_safe(score_scl)

        def _is_saturated(v):
            return v <= 0.1 or v >= 99.9

        if _is_saturated(pct_raw) and not _is_saturated(pct_scl):
            chosen = pct_scl
            chosen_src = "scaled"
            chosen_score = score_scl
        else:
            chosen = pct_raw
            chosen_src = "raw"
            chosen_score = score_raw

        preds[metric] = chosen
        _debug[metric] = {
            "source": chosen_src,
            "model_output": chosen_score,
            "postproc": "to_percent_safe",
            "raw_score": score_raw,
            "scaled_score": score_scl,
            "raw%": pct_raw,
            "scaled%": pct_scl,
            "display": chosen,
        }

st.session_state["_model_debug"] = _debug

# Apply comfort baseline floor (kept from your previous tool behavior)
display_preds = preds.copy()
for _m in ["PMV", "PPD"]:
    _base = BASELINES.get(_m)
    if _base is not None and _m in display_preds:
        if float(display_preds[_m]) < float(_base):
            display_preds[_m] = float(_base)
            if _m in _debug:
                _debug[_m]["baseline_floor"] = float(_base)
                _debug[_m]["display"] = float(_base)
                _debug[_m]["postproc"] = (_debug[_m].get("postproc", "") + " + baseline_floor").strip()

# If baseline config selected, force displayed scenario to BASELINES
# (does NOT affect facades / other retrofit cases)
if is_baseline_case(wall_type, roof_type, window_type):
    for k, v in BASELINES.items():
        if k in display_preds:
            display_preds[k] = float(v)
            if k in _debug:
                _debug[k]["display_forced_to_baseline"] = True

# ────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────
st.title("BIO4EEB PERFORMANCE TOOL — MULTI-METRIC (Batch-ready)")
st.caption("Scaled RF for EUI/CD/HD; probability % for PMV/PPD; batch upload supported.")

units_short = {"EUI": "kWh/m²·yr", "CD": "kWh/m²", "HD": "kWh/m²", "PMV": "%", "PPD": "%"}

# ENERGY KPIs
st.subheader("Energy KPIs (lower is better)")
st.markdown("""
- **EUI**: Energy use per floor area per year.
- **CD**: Cooling demand (energy for cooling).
- **HD**: Heating demand (energy for heating).
""")

row1 = st.columns(3)
for col, m in zip(row1, ["EUI", "CD", "HD"]):
    v = float(display_preds[m])
    base = float(BASELINES[m])
    # Show Scenario − Baseline. Negative is good => delta_color inverse makes negative green.
    delta = v - base
    col.metric(
        f"{m} ({units_short[m]})",
        f"{v:.2f}",
        delta=f"{delta:+.2f} vs baseline",
        delta_color="inverse",
    )

def render_two_bar_energy(metric_key: str):
    base = float(BASELINES.get(metric_key, 0.0))
    val = float(display_preds.get(metric_key, 0.0))
    ymax = (max(base, val) * 1.25) if max(base, val) > 0 else 1.0

    improvement = base - val  # positive = better (lower than baseline)
    improvement_pct = (improvement / base * 100.0) if base > 0 else 0.0

    label_map = {
        "EUI": "EUI: Energy use per floor area per year.",
        "CD":  "CD: Cooling demand (energy for cooling).",
        "HD":  "HD: Heating demand (energy for heating).",
    }
    disp_label = label_map.get(metric_key, metric_key)

    fig = go.Figure()
    color_map = {"EUI": "#FFA500", "CD": "#1976d2", "HD": "#e53935"}
    scenario_color = color_map.get(metric_key, "#1976d2")

    fig.add_bar(name="Baseline", x=[disp_label], y=[base], marker_color="#9e9e9e",
                text=[f"{base:.2f}"], textposition="outside")
    fig.add_bar(name="Scenario", x=[disp_label], y=[val], marker_color=scenario_color,
                text=[f"{val:.2f}"], textposition="outside")

    arrow = "↓" if improvement > 0 else ("↑" if improvement < 0 else "→")
    txt = f"{arrow} {abs(improvement):.2f} {units_short[metric_key]} ({abs(improvement_pct):.1f}%) vs baseline"
    fig.add_annotation(
        x=disp_label, y=ymax * 0.95, text=txt, showarrow=False,
        font=dict(size=11, color="#2e7d32" if improvement > 0 else "#c62828" if improvement < 0 else "#616161"),
        xanchor="center", yanchor="top"
    )

    fig.update_layout(
        barmode="group",
        yaxis_title=units_short[metric_key],
        yaxis_range=[0, ymax],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

for _m in ["EUI", "CD", "HD"]:
    render_two_bar_energy(_m)

ref_txt = st.session_state.get("_refurb_col_used") or "None"
st.caption(
    f"Wall: {wall_type}  •  Roof: {roof_type}  •  Window: {window_type}  •  "
    f"Refurbishment col: {ref_txt}"
)

# COMFORT KPIs
st.subheader("Comfort KPIs (higher is better)")
st.markdown("""
- **PMV**: Percentage of time occupants feel neutral (PMV within ±0.5).
- **PPD**: Percentage of time with less than 10% of people dissatisfied.
""")

row2 = st.columns(2)
for col, m in zip(row2, ["PMV", "PPD"]):
    v = float(display_preds[m])
    base = float(BASELINES[m])
    delta = v - base  # positive is good
    col.metric(f"{m} ({units_short[m]})", f"{v:.2f}", delta=f"{delta:+.2f} vs baseline")

# ────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ────────────────────────────────────────────────────────────────────────────
if show_diag:
    dbg = st.session_state.get("_model_debug", {})
    rows = []
    for m in ["EUI", "CD", "HD", "PMV", "PPD"]:
        if m in dbg and m in preds:
            r = dbg[m]
            rows.append({
                "Metric": m,
                "Source": r.get("source", "scaled" if m in {"EUI","CD","HD"} else "raw"),
                "Model output": r.get("model_output", np.nan),
                "Displayed": display_preds.get(m, preds[m]),
                "Post-processing": r.get("postproc", "identity"),
            })
    if rows:
        ver_df = pd.DataFrame(rows)
        st.subheader("Diagnostics")
        st.dataframe(ver_df, use_container_width=True)

        st.caption(
            f"Wall one-hot: {st.session_state.get('_onehot_wall_col_used')}  •  "
            f"Roof one-hot: {st.session_state.get('_onehot_roof_col_used')}  •  "
            f"Window one-hot: {st.session_state.get('_onehot_window_col_used')}"
        )
        mf = st.session_state.get("_missing_flags", [])
        if mf:
            st.warning("\n".join(mf))

        st.download_button(
            "Download diagnostics (CSV)",
            data=ver_df.to_csv(index=False).encode("utf-8"),
            file_name="bio4eeb_diagnostics.csv",
            mime="text/csv"
        )

# ────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
st.subheader("Batch predictions from file")
st.markdown("Upload either **(A)** a simple sheet with high-level inputs (columns below), or **(B)** a sheet that already matches the model schema (all expected features).")

example_df = pd.DataFrame([{
    "wall_type": "EPS",
    "wall_thickness_m": 0.10,
    "roof_type": "PLA foam",
    "roof_thickness_m": 0.10,
    "window_type": "Window_BIO4EEB W2",
}])
st.download_button(
    "Download simple template (CSV)",
    data=example_df.to_csv(index=False).encode("utf-8"),
    file_name="bio4eeb_input_template.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

def canonical_material(label: str, allowed: list[str]) -> Optional[str]:
    n = norm_key(label)
    # direct match
    for a in allowed:
        if norm_key(a) == n:
            return a
    # allow common facade variants from file inputs
    # e.g., "Facade_type 5", "Facade Type 5", "Facade_Type5", etc.
    if "facade" in n:
        digits = "".join(ch for ch in str(label) if ch.isdigit())
        if digits:
            for a in allowed:
                if a in FACADE_TYPES and digits in a:
                    return a
    return None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith("csv"):
            user_df = pd.read_csv(uploaded)
        else:
            user_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        user_df = None

    if user_df is not None:
        st.write(f"Loaded file with shape: {user_df.shape}")

        baseline_flags = None

        # Case B: full feature matrix
        if set(EXPECTED_COLS).issubset(set(user_df.columns)):
            st.info("Detected full feature matrix. Using it as-is.")
            raw_mat = user_df[EXPECTED_COLS].copy()
            scaled_mat = pd.DataFrame(scaler.transform(raw_mat), columns=EXPECTED_COLS)

        else:
            # Case A: simple inputs
            required = {"wall_type", "wall_thickness_m", "roof_type", "roof_thickness_m", "window_type"}
            missing = required - set(map(str, user_df.columns))
            if missing:
                st.error(f"Missing required columns for simple mode: {sorted(list(missing))}")
                st.stop()

            rows = []
            baseline_flags = []

            for _, r in user_df.iterrows():
                wt_in = str(r["wall_type"]).strip()
                rt_in = str(r["roof_type"]).strip()
                win_in = str(r["window_type"]).strip()

                wt = canonical_material(wt_in, list(INSULATION_PROPERTIES.keys()))
                rt = canonical_material(rt_in, [k for k in INSULATION_PROPERTIES.keys() if k not in set(FACADE_TYPES)])
                win = canonical_material(win_in, list(WINDOW_PROPERTIES.keys()))

                if wt is None:
                    st.warning(f"Unknown wall_type '{wt_in}' → using 'No_insulation'")
                    wt = "No_insulation"
                if rt is None:
                    st.warning(f"Unknown roof_type '{rt_in}' → using 'No_insulation'")
                    rt = "No_insulation"
                if win is None:
                    st.warning(f"Unknown window_type '{win_in}' → using 'Window_Existing'")
                    win = "Window_Existing"

                wth = float(r["wall_thickness_m"]) if pd.notna(r["wall_thickness_m"]) else 0.0
                rth = float(r["roof_thickness_m"]) if pd.notna(r["roof_thickness_m"]) else 0.0

                # Facade thickness always taken from trained values
                if wt in FACADE_TYPES:
                    wth = float(FACADE_TRAINED_THICKNESS[wt])

                # Baseline flags for forcing baseline display on baseline config only
                baseline_flags.append(is_baseline_case(wt, rt, win))

                rows.append(build_feature_row(wt, wth, rt, rth, win, enable_proxy=True))

            raw_mat = pd.DataFrame(rows, columns=EXPECTED_COLS)
            scaled_mat = pd.DataFrame(scaler.transform(raw_mat), columns=EXPECTED_COLS)

        # Predict
        out = pd.DataFrame()
        for metric, mdl in models.items():
            if hasattr(mdl, "feature_names_in_"):
                X_raw = raw_mat[list(mdl.feature_names_in_)]
                X_scl = scaled_mat[list(mdl.feature_names_in_)]
            else:
                X_raw = raw_mat.to_numpy()
                X_scl = scaled_mat.to_numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if metric in {"EUI", "CD", "HD"}:
                    out[metric] = mdl.predict(X_scl)
                else:
                    y_raw = mdl.predict(X_raw)
                    y_scl = mdl.predict(X_scl)
                    pct_raw = np.array([to_percent_safe(float(v)) for v in y_raw])
                    pct_scl = np.array([to_percent_safe(float(v)) for v in y_scl])
                    sat_raw = (pct_raw <= 0.1) | (pct_raw >= 99.9)
                    sat_scl = (pct_scl <= 0.1) | (pct_scl >= 99.9)
                    choose_scl = sat_raw & ~sat_scl
                    out[metric] = np.where(choose_scl, pct_scl, pct_raw)

        # Comfort baseline floor
        for _m in ["PMV", "PPD"]:
            if _m in out.columns:
                out[_m] = np.maximum(out[_m], BASELINES[_m])

        # Force baseline rows to baseline values (only for true baseline configuration)
        if baseline_flags is not None:
            flags = np.array(baseline_flags, dtype=bool)
            for m, base in BASELINES.items():
                if m in out.columns:
                    out.loc[flags, m] = float(base)

        # Deltas
        for m, base in BASELINES.items():
            if m in out.columns:
                if m in {"PMV", "PPD"}:
                    out[f"Δ {m} vs baseline"] = out[m] - base
                else:
                    out[f"Δ {m} vs baseline"] = base - out[m]

        for _m in ["PMV", "PPD"]:
            _col = f"Δ {_m} vs baseline"
            if _col in out.columns:
                out[_col] = np.maximum(out[_col], 0.0)

        st.success("Predictions ready.")
        display_df = out.copy()
        for c in display_df.columns:
            if c in {"PMV", "PPD", "EUI", "CD", "HD"} or c.startswith("Δ "):
                display_df[c] = np.round(display_df[c].astype(float), 2)

        st.dataframe(display_df.head(50), use_container_width=True)
        st.download_button(
            "Download predictions (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="bio4eeb_batch_predictions.csv",
            mime="text/csv"
        )

# ────────────────────────────────────────────────────────────────────────────
# Model insights — RF feature importances
# ────────────────────────────────────────────────────────────────────────────
with st.expander("Model insights: feature importance (RF)"):
    for m in ["EUI", "CD", "HD"]:
        mdl = models.get(m)
        if mdl is not None and hasattr(mdl, "feature_importances_"):
            imp = pd.Series(mdl.feature_importances_, index=getattr(mdl, "feature_names_in_", EXPECTED_COLS))
            top = imp.sort_values(ascending=False).head(8)
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=top.values, y=top.index, ax=ax)
            ax.set_title(f"Top drivers for {m}")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        else:
            st.caption(f"{m}: importance not available (model is not RF or lacks attribute).")

st.markdown("""
---
**Notes**  
• Facade types are available under **Wall insulation type** only; roof excludes facades.  
• Facade thickness is set to the **trained thickness values** (per Hungary_update_final_FacadeV00.ipynb).  
• Baseline chart values come from BASELINES.  
• Baseline configuration (No insulation / No insulation / Window_Existing) is forced to match BASELINES for display.  
""", unsafe_allow_html=True)
