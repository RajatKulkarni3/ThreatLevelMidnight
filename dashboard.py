"""Read-only dark SaaS dashboard for exported fraud-spike detection results."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Fraud Spike Detector", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #0B1120; color: #FFFFFF; }
      [data-testid="stMetric"] {
        background: #141B2D;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
      }
      [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] { color: #9CA3AF; }
      [data-testid="stMetricValue"] { color: #FFFFFF; }
      h1, h2, h3 { color: #FFFFFF; }
      .teal-accent { color: #1D9E75; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fraud spike detector")
st.caption("Real-time card-testing detection, distinguished from legitimate demand spikes")

results_path = Path("results.json")
if not results_path.exists():
    st.error("Missing results.json. Run `python fraud_spike_mvp.py` first.")
    st.stop()

with results_path.open(encoding="utf-8") as result_file:
    results = json.load(result_file)

velocity = pd.DataFrame(results["velocity_series"])
velocity["window_start"] = pd.to_datetime(velocity["window_start"])
merchants = sorted(velocity["merchant_id"].unique())
selected_merchant = st.selectbox("Merchant", merchants)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout attacks detected", results["attacks_detected"])
col2.metric("Median latency", f"{results['median_latency_min']:.2f} min")
col3.metric("Precision", f"{results['precision']:.2f}")
col4.metric("False-positive rate", f"{results['fp_rate_all_nonattack']:.1%}")

merchant_velocity = velocity[velocity["merchant_id"] == selected_merchant]
merchant_alerts = [alert for alert in results["alerts"] if alert["merchant_id"] == selected_merchant]
merchant_injections = [item for item in results["injections"] if item["merchant_id"] == selected_merchant]

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=merchant_velocity["window_start"], y=merchant_velocity["velocity"],
    mode="lines", name="Velocity", line={"color": "#1D9E75", "width": 2},
    hovertemplate="%{x}<br>Velocity: %{y:.2f} txns/sec<extra></extra>",
))
seen_labels = set()
for injection in merchant_injections:
    color = "#EF4444" if injection["label"] == "card_testing" else "#22C55E"
    label = "True attack start" if injection["label"] == "card_testing" else "True demand-spike start"
    figure.add_vline(x=pd.to_datetime(injection["start"]), line_color=color, line_dash="dash", line_width=1.5)
    if label not in seen_labels:
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label,
                                    line={"color": color, "dash": "dash"}))
        seen_labels.add(label)
if merchant_alerts:
    alert_times = pd.to_datetime([alert["window_start"] for alert in merchant_alerts])
    alert_velocity = merchant_velocity.set_index("window_start")["velocity"].reindex(alert_times).to_numpy()
    figure.add_trace(go.Scatter(
        x=alert_times, y=alert_velocity, mode="markers", name="Alert fired",
        marker={"color": "#000000", "symbol": "triangle-up", "size": 11, "line": {"color": "#FFFFFF", "width": 1}},
        hovertemplate="%{x}<br>Card-testing alert<extra></extra>",
    ))
figure.update_layout(
    template="plotly_dark", height=440, margin={"l": 12, "r": 12, "t": 45, "b": 12},
    title=f"{selected_merchant}: transaction velocity", paper_bgcolor="#0B1120", plot_bgcolor="#141B2D",
    xaxis_title="Window start", yaxis_title="Transactions / second", legend={"orientation": "h", "y": 1.12},
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Highest-confidence alerts")
for alert in sorted(merchant_alerts, key=lambda item: item["confidence"], reverse=True)[:5]:
    st.markdown(
        f"""
        <div style="background:#141B2D;border:1px solid #1F2937;border-left:4px solid #1D9E75;
                    border-radius:8px;padding:16px;margin-bottom:12px;">
          <div style="color:#FFFFFF;font-weight:600;font-size:16px;">
            Fraud spike detected — {alert['merchant_id']}
          </div>
          <div style="color:#9CA3AF;font-size:13px;margin-top:4px;">
            Confidence: {alert['confidence'] * 100:.1f}% · Window: {alert['window_start']}
          </div>
          <div style="color:#D1D5DB;font-size:13px;margin-top:8px;">
            Pattern: card testing — narrow BIN range, high decline rate
          </div>
          <div style="color:#1D9E75;font-size:13px;margin-top:6px;font-weight:500;">
            Suggested action: temporarily hold approvals for the flagged BIN range
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

counts = results["test_window_counts"]
st.caption(
    f"Evaluation: {results['attacks_detected']} holdout attacks detected; {counts['card_testing']} card-testing windows, "
    f"{counts['demand_spike']} demand-spike windows, and {counts['quiet_period']} quiet-period windows. "
    "Synthetic evaluation demonstrates the detection mechanism, not a production-scale false-positive rate."
)
