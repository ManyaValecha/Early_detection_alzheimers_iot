"""
3_Dementia_Monitoring.py
Longitudinal in-home IoT sensor telemonitoring for People Living with Dementia (PLWD).
Visualizes multi-modal telemetry streams: PIR motion, sleep disturbance, vital signs, and predicted alerts.
"""

import os
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
# Ensure robust multi-environment path resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from feature_engineering import extract_tihm_monitoring_features
except ImportError:
    from aide_tihm.src.feature_engineering import extract_tihm_monitoring_features

st.set_page_config(page_title="Dementia IoT Monitoring", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; }
    .alert-banner-danger { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 12px 18px; border-radius: 6px; margin-bottom: 12px; }
    .alert-banner-warn { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 12px 18px; border-radius: 6px; margin-bottom: 12px; }
    .alert-banner-ok { background-color: #F0FDF4; border-left: 5px solid #10B981; padding: 12px 18px; border-radius: 6px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📡 Longitudinal Dementia Telemonitoring (TIHM IoT Cohort)</div>', unsafe_allow_html=True)
st.caption("Continuous In-Home Passive Telemetry: Activity Entropy, Sleep Restlessness, Vitals & Acute Event Prediction")

st.markdown("---")

# Load TIHM data and trained Model 2
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "tihm"))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

tihm_csv_path = os.path.join(data_dir, "tihm_longitudinal_merged.csv")
model2_path = os.path.join(models_dir, "tihm_event_model.joblib")
scaler2_path = os.path.join(models_dir, "tihm_scaler.joblib")
feats2_path = os.path.join(models_dir, "tihm_features.json")

if not os.path.exists(tihm_csv_path):
    st.error("TIHM dataset not found. Please run the training pipeline (`python src/train_pipeline.py`) to initialize datasets.")
    st.stop()

df_tihm = pd.read_csv(tihm_csv_path)

# Participant selector in sidebar
st.sidebar.header("Participant Selector")
patient_list = sorted(df_tihm["patient_id"].unique())
selected_pid = st.sidebar.selectbox("Select Monitored Participant:", patient_list, index=0)

df_p = df_tihm[df_tihm["patient_id"] == selected_pid].sort_values("date").reset_index(drop=True)

# Participant Demographics Header
demo_age = df_p["age"].iloc[0]
demo_gender = df_p["gender"].iloc[0]
demo_diag = df_p["diagnosis"].iloc[0]
demo_liv = df_p["living_status"].iloc[0]

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
col_d1.metric("Participant ID", selected_pid)
col_d2.metric("Age / Gender", f"{demo_age} yrs / {demo_gender}")
col_d3.metric("Clinical Diagnosis", demo_diag)
col_d4.metric("Household Context", demo_liv)

st.markdown("---")

# Run Model 2 predictions on participant trajectory if model exists
model2_available = os.path.exists(model2_path) and os.path.exists(scaler2_path)
if model2_available:
    clf2 = joblib.load(model2_path)
    scaler2 = joblib.load(scaler2_path)
    X_p, y_p, _, _, df_p_proc = extract_tihm_monitoring_features(df_p, is_training=False, scaler=scaler2)
    df_p["predicted_event_prob"] = clf2.predict_proba(X_p)[:, 1]
else:
    df_p["predicted_event_prob"] = 0.0

# Alert summary
n_events = int(df_p["health_event_flag"].sum())
recent_event = df_p.tail(3)["health_event_flag"].sum() > 0
recent_prob = df_p["predicted_event_prob"].iloc[-1]

if recent_prob >= 0.50 or recent_event:
    recent_type = df_p[df_p["health_event_flag"] == 1]["event_type"].iloc[-1] if n_events > 0 else "Anomalous Deviation"
    st.markdown(f"""
    <div class="alert-banner-danger">
        <strong>⚠️ CLINICAL TELEMONITORING ALERT ACTIVE:</strong>
        Elevated risk probability (<strong>{recent_prob:.1%}</strong>) detected on {df_p['date'].iloc[-1]}.
        Recent event signature: <strong>{recent_type}</strong>.
        Action: Clinical triage nurse dispatched for verification.
    </div>
    """, unsafe_allow_html=True)
elif recent_prob >= 0.30:
    st.markdown(f"""
    <div class="alert-banner-warn">
        <strong>⚠️ PRE-SYMPTOMATIC DEVIATION DETECTED:</strong>
        Mild sensor anomaly score (<strong>{recent_prob:.1%}</strong>). Elevated nocturnal restlessness and bathroom visits.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-banner-ok">
        <strong>✅ TELEMETRY STABLE:</strong>
        Participant baseline is within normal circadian and physiological parameters (Current risk probability: <strong>{recent_prob:.1%}</strong>).
    </div>
    """, unsafe_allow_html=True)

# Tabs for Telemetry Streams
tab_act, tab_sleep, tab_vitals, tab_alerts = st.tabs([
    "🚶 In-Home Activity & Rooms",
    "🛌 Sleep Patterns & Restlessness",
    "💓 Physiological Telemetry",
    "🚨 Predictive Health-Event Signals"
])

with tab_act:
    st.subheader("Daily Motion Counts by Zone (PIR Sensors)")
    fig_act = go.Figure()
    fig_act.add_trace(go.Bar(x=df_p["date"], y=df_p["pir_bedroom"], name="Bedroom", marker_color="#3B82F6"))
    fig_act.add_trace(go.Bar(x=df_p["date"], y=df_p["pir_living"], name="Living Room", marker_color="#10B981"))
    fig_act.add_trace(go.Bar(x=df_p["date"], y=df_p["pir_kitchen"], name="Kitchen", marker_color="#F59E0B"))
    fig_act.add_trace(go.Bar(x=df_p["date"], y=df_p["pir_bathroom"], name="Bathroom", marker_color="#EC4899"))
    fig_act.add_trace(go.Bar(x=df_p["date"], y=df_p["pir_hallway"], name="Hallway", marker_color="#8B5CF6"))

    fig_act.update_layout(
        barmode="stack",
        height=380,
        xaxis_title="Date",
        yaxis_title="PIR Motion Activations",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_act, use_container_width=True)

    st.subheader("Room Transitions & Nocturnal Bathroom Intensity")
    fig_trans = go.Figure()
    fig_trans.add_trace(go.Scatter(x=df_p["date"], y=df_p["room_transitions"], mode="lines+markers", name="Daily Room Transitions", line=dict(color="#1E3A8A", width=2)))
    fig_trans.add_trace(go.Bar(x=df_p["date"], y=df_p["night_bathroom_trips"], name="Night Bathroom Trips", yaxis="y2", marker_color="#EF4444", opacity=0.6))

    fig_trans.update_layout(
        height=320,
        xaxis_title="Date",
        yaxis=dict(title="Room Transitions"),
        yaxis2=dict(title="Night Bathroom Trips", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trans, use_container_width=True)

with tab_sleep:
    st.subheader("Under-Mattress Sensor: Sleep Duration & Sleep Efficiency")
    fig_sleep = go.Figure()
    fig_sleep.add_trace(go.Bar(x=df_p["date"], y=df_p["sleep_duration_hours"], name="Sleep Hours", marker_color="#6366F1"))
    fig_sleep.add_trace(go.Scatter(x=df_p["date"], y=df_p["sleep_efficiency"], name="Sleep Efficiency (%)", yaxis="y2", line=dict(color="#10B981", width=2.5)))

    fig_sleep.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis=dict(title="Sleep Hours", range=[0, 12]),
        yaxis2=dict(title="Efficiency (%)", overlaying="y", side="right", range=[30, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sleep, use_container_width=True)

    st.subheader("Nocturnal Restlessness Score & Bed Exit Count")
    fig_rest = go.Figure()
    fig_rest.add_trace(go.Scatter(x=df_p["date"], y=df_p["sleep_restlessness_score"], mode="lines+markers", name="Restlessness Index", line=dict(color="#D97706", width=2)))
    fig_rest.add_trace(go.Bar(x=df_p["date"], y=df_p["bed_exit_count"], name="Bed Exits", yaxis="y2", marker_color="#DC2626", opacity=0.5))

    fig_rest.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis=dict(title="Restlessness Score (0-1)"),
        yaxis2=dict(title="Bed Exits", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_rest, use_container_width=True)

with tab_vitals:
    st.subheader("Longitudinal Blood Pressure & Resting Pulse Rate")
    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(x=df_p["date"], y=df_p["systolic_bp"], name="Systolic BP (mmHg)", line=dict(color="#DC2626", width=2)))
    fig_bp.add_trace(go.Scatter(x=df_p["date"], y=df_p["diastolic_bp"], name="Diastolic BP (mmHg)", line=dict(color="#F59E0B", width=2)))
    fig_bp.add_trace(go.Scatter(x=df_p["date"], y=df_p["pulse"], name="Resting Pulse (bpm)", yaxis="y2", line=dict(color="#2563EB", width=2, dash="dot")))

    fig_bp.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis=dict(title="Blood Pressure (mmHg)"),
        yaxis2=dict(title="Heart Rate (bpm)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bp, use_container_width=True)

with tab_alerts:
    st.subheader("Model 2 Telemonitoring Risk Trajectory vs Clinical Ground Truth")
    fig_event = go.Figure()

    fig_event.add_trace(go.Scatter(
        x=df_p["date"],
        y=df_p["predicted_event_prob"],
        mode="lines+markers",
        name="Model Event Risk Probability",
        line=dict(color="#7C3AED", width=2.5)
    ))

    # Mark true events
    true_event_days = df_p[df_p["health_event_flag"] == 1]
    if len(true_event_days) > 0:
        fig_event.add_trace(go.Scatter(
            x=true_event_days["date"],
            y=true_event_days["predicted_event_prob"],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="red"),
            text=true_event_days["event_type"],
            textposition="top center",
            name="Clinician-Confirmed Event"
        ))

    # Threshold line
    fig_event.add_hline(y=0.50, line_dash="dash", line_color="red", annotation_text="Triage Alert Threshold (0.50)")

    fig_event.update_layout(
        height=360,
        xaxis_title="Date",
        yaxis=dict(title="Adverse Event Probability", range=[0.0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_event, use_container_width=True)

    st.markdown(r"""
    **Clinical Telehealth Workflow:**
    When the multi-modal anomaly detector scores probability $\ge 0.50$, an automated care-coordinator notification is triggered.
    For instance, nocturnal bathroom trip surges combined with reduced sleep duration and temperature elevation flag suspected **Urinary Tract Infections (UTIs)**,
    enabling non-invasive antibiotic intervention before delirium or septic emergency occurs.
    """)
