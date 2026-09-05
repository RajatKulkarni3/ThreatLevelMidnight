"""Read-only dark SaaS dashboard for exported fraud-spike detection results."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="ThreatLevelMidnight — Fraud Spike Detector", layout="wide")
st.markdown(
    """
    <style>
      html, body, .stApp,
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stHeader"],
      .main {
        background-color: #0B1120 !important;
        color: #FFFFFF !important;
      }
      #MainMenu, footer, header { visibility: hidden; }

      [data-testid="stMetric"] {
        background: #141B2D;
        border: 1px solid #1F2937;
        border-top: 3px solid #1D9E75;
        border-radius: 12px;
        padding: 16px;
      }
      [data-testid="stMetricLabel"] { color: #9CA3AF !important; text-transform: uppercase; letter-spacing: 0.4px; font-size: 11px; }
      [data-testid="stCaptionContainer"] { color: #9CA3AF !important; }
      [data-testid="stMetricValue"] { color: #2BD99C !important; font-weight: 700; }
      h1, h2, h3 { color: #FFFFFF; }
      .teal-accent { color: #1D9E75; }

      /* Top header bar, matching the buildathon concept video */
      .tlm-header {
        display: flex; align-items: center; justify-content: space-between;
        border-bottom: 1px solid #1F2937; padding-bottom: 14px; margin-bottom: 22px;
      }
      .tlm-logo { display: flex; align-items: center; gap: 10px; }
      .tlm-dot { width: 10px; height: 10px; border-radius: 3px; background: #2BD99C; flex-shrink: 0; }
      .tlm-wordmark { font-size: 22px; font-weight: 800; color: #FFFFFF !important; letter-spacing: 0.2px; }
      .tlm-wordmark span { color: #2BD99C !important; }
      .tlm-sub { font-size: 12.5px; color: #9CA3AF !important; margin-top: 2px; }
      .tlm-badge {
        font-size: 11px; color: #9CA3AF !important; border: 1px solid #1F2937;
        padding: 6px 12px; border-radius: 20px; white-space: nowrap;
      }

      /* Merchant tag pill next to the selector */
      .tlm-merchant-tag {
        display: inline-block; font-size: 11px; color: #2BD99C !important;
        background: rgba(29,158,117,0.12); border: 1px solid rgba(29,158,117,0.35);
        padding: 4px 11px; border-radius: 20px; margin-top: 6px;
      }

      /* Pattern tag pills inside alert cards */
      .tlm-tagrow { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
      .tlm-tag {
        font-size: 11px; background: rgba(239,68,68,0.12); color: #FCA5A5 !important;
        border: 1px solid rgba(239,68,68,0.3); padding: 4px 10px; border-radius: 20px;
      }
      .tlm-pattern-label {
        font-size: 10.5px; color: #6B7280 !important; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 14px;
      }

      /* Live feed panel: the featured alert + pulsing indicator + slide-in motion */
      .tlm-live-label {
        display: flex; align-items: center; gap: 7px; font-size: 11px;
        color: #F87171 !important; text-transform: uppercase; letter-spacing: 0.6px;
        font-weight: 700; margin-bottom: 10px;
      }
      .tlm-pulse-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #EF4444;
        box-shadow: 0 0 0 0 rgba(239,68,68,0.6);
        animation: tlm-pulse 1.6s infinite;
      }
      @keyframes tlm-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.55); }
        70%  { box-shadow: 0 0 0 9px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
      }
      .tlm-alert-featured {
        animation: tlm-slidein 0.45s ease-out;
      }
      @keyframes tlm-slidein {
        from { opacity: 0; transform: translateX(24px); }
        to   { opacity: 1; transform: translateX(0); }
      }
      .tlm-stat-chip-row { display: flex; gap: 8px; margin-top: 12px; }
      .tlm-stat-chip {
        flex: 1; background: #141B2D; border: 1px solid #1F2937; border-radius: 10px;
        padding: 10px 8px; text-align: center;
      }
      .tlm-stat-chip .lbl { font-size: 9px; color: #9CA3AF !important; text-transform: uppercase; letter-spacing: 0.4px; }
      .tlm-stat-chip .val { font-size: 18px; font-weight: 700; color: #2BD99C !important; margin-top: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tlm-header">
      <div>
        <div class="tlm-logo">
          <div class="tlm-dot"></div>
          <div class="tlm-wordmark">Threat<span>Level</span>Midnight</div>
        </div>
        <div class="tlm-sub">Card-testing detection on held-out synthetic transaction windows, distinguished from legitimate demand spikes</div>
      </div>
      <div class="tlm-badge">Track 2 · AI Risk Manager · Razorpay AI Buildathon</div>
    </div>
    """,
    unsafe_allow_html=True,
)

results_path = Path("results.json")
if not results_path.exists():
    st.error("Missing results.json. Run `python fraud_spike_mvp.py` first.")
    st.stop()

with results_path.open(encoding="utf-8") as result_file:
    results = json.load(result_file)

aggregate = results.get("aggregate_evaluation")
notes = results.get("methodology_notes", {})

velocity = pd.DataFrame(results["velocity_series"])
velocity["window_start"] = pd.to_datetime(velocity["window_start"])
merchants = sorted(velocity["merchant_id"].unique())
selected_merchant = st.selectbox("Merchant", merchants)
st.markdown(f'<div class="tlm-merchant-tag">{selected_merchant}</div>', unsafe_allow_html=True)

st.subheader(f"Primary run (seed {results.get('primary_seed', '—')})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout attacks detected", results["attacks_detected"])
col2.metric("Median latency", results.get("median_latency_display", f"{results['median_latency_min']:.2f} min"))
col3.metric("Precision", f"{results['precision']:.2f}")
col4.metric("False-positive rate", f"{results['fp_rate_all_nonattack']:.1%}")
st.caption(notes.get(
    "latency_interpretation",
    f"Detection latency is measured at the simulation/window resolution "
    f"({results.get('window_seconds', 30)}-second windows) and is not a production "
    "network-latency benchmark.",
))

if aggregate:
    st.subheader(f"Aggregate across {aggregate['n_seeds']} independent synthetic scenarios")
    st.caption("Same pipeline, freshly generated synthetic data each time — shows how much the metrics move, not just one run.")
    acol1, acol2, acol3, acol4 = st.columns(4)
    acol1.metric("Precision (mean±std)", f"{aggregate['precision']['mean']:.2f} ± {aggregate['precision']['std']:.2f}")
    acol2.metric("Recall (mean±std)", f"{aggregate['recall']['mean']:.2f} ± {aggregate['recall']['std']:.2f}")
    acol3.metric("F1 (mean±std)", f"{aggregate['f1']['mean']:.2f} ± {aggregate['f1']['std']:.2f}")
    acol4.metric(
        "FP rate (mean±std)",
        f"{aggregate['fp_rate_all_nonattack']['mean']:.1%} ± {aggregate['fp_rate_all_nonattack']['std']:.1%}",
    )
    with st.expander("Per-seed breakdown"):
        st.dataframe(pd.DataFrame(aggregate["per_seed"]), use_container_width=True, hide_index=True)

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
    template="plotly_dark", height=460, margin={"l": 12, "r": 12, "t": 50, "b": 90},
    title={"text": f"{selected_merchant}: transaction velocity", "y": 0.97},
    paper_bgcolor="#0B1120", plot_bgcolor="#141B2D",
    xaxis_title="Window start", yaxis_title="Transactions / second",
    legend={"orientation": "h", "y": -0.22, "yanchor": "top", "x": 0.5, "xanchor": "center"},
)


def describe_pattern(feature_values: dict) -> list[str]:
    """Build an honest, alert-specific set of tags from the actual feature
    values instead of a single hardcoded pattern description — different
    attack variants (narrow vs. distributed, etc.) look quite different."""
    tags = []
    decline = feature_values.get("decline_rate", 0)
    if decline > 0.5:
        tags.append(f"{decline:.0%} decline rate")
    elif decline > 0.25:
        tags.append(f"elevated decline rate ({decline:.0%})")
    if feature_values.get("device_diversity_ratio", 1) < 0.3:
        tags.append("low device diversity")
    if feature_values.get("ip_diversity_ratio", 1) < 0.3:
        tags.append("low IP diversity")
    if feature_values.get("bin_diversity_ratio", 1) < 0.3:
        tags.append("narrow BIN pool")
    if feature_values.get("pct_low_amount", 0) > 0.6:
        tags.append(f"{feature_values['pct_low_amount']:.0%} low-value txns")
    return tags if tags else ["moderate anomaly signature across several features"]


def render_alert_card(alert: dict, featured: bool = False) -> None:
    tags = describe_pattern(alert.get("features", {}))
    tag_html = "".join(f'<span class="tlm-tag">{tag}</span>' for tag in tags)
    wrapper_class = "tlm-alert-featured" if featured else ""
    st.markdown(
        f"""
        <div class="{wrapper_class}" style="background:#141B2D;border:1px solid #1F2937;border-left:4px solid #EF4444;
                    border-radius:10px;padding:18px;margin-bottom:14px;">
          <div style="color:#FFFFFF;font-weight:700;font-size:16px;">
            Fraud spike detected — {alert['merchant_id']}
          </div>
          <div style="color:#9CA3AF;font-size:12.5px;margin-top:5px;">
            Model score: {alert['model_score'] * 100:.1f}% (uncalibrated) · Window: {alert['window_start']}
          </div>
          <div class="tlm-pattern-label">Pattern — card testing</div>
          <div class="tlm-tagrow">{tag_html}</div>
          <div style="margin-top:14px;padding:11px 12px;background:rgba(29,158,117,0.1);
                      border:1px solid rgba(29,158,117,0.3);border-radius:8px;
                      color:#2BD99C;font-size:12.5px;">
            Suggested action: escalate the flagged activity for additional review. Any payment
            intervention should remain subject to merchant risk policy and human approval.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


sorted_alerts = sorted(merchant_alerts, key=lambda item: item["model_score"], reverse=True)

chart_col, live_col = st.columns([2, 1])

with chart_col:
    st.plotly_chart(figure, use_container_width=True)

with live_col:
    st.markdown(
        '<div class="tlm-live-label"><span class="tlm-pulse-dot"></span>Held-out alert feed</div>',
        unsafe_allow_html=True,
    )
    st.caption(notes.get(
        "score_interpretation",
        "Model score is the classifier's own probability estimate, not a calibrated real-world fraud likelihood.",
    ))
    if sorted_alerts:
        render_alert_card(sorted_alerts[0], featured=True)
        st.markdown(
            f"""
            <div class="tlm-stat-chip-row">
              <div class="tlm-stat-chip"><div class="lbl">Precision</div><div class="val">{results['precision']*100:.1f}%</div></div>
              <div class="tlm-stat-chip"><div class="lbl">Recall</div><div class="val">{results['recall']*100:.1f}%</div></div>
              <div class="tlm-stat-chip"><div class="lbl">Detected in</div><div class="val">{results.get('median_latency_display', f"{results['median_latency_min']:.1f} min")}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if len(sorted_alerts) > 1:
            with st.expander(f"Show {len(sorted_alerts) - 1} more alerts for {selected_merchant}"):
                for alert in sorted_alerts[1:6]:
                    render_alert_card(alert)
    else:
        st.info("No card-testing alerts fired for this merchant.")

fp_impact = results.get("false_positive_impact")
if fp_impact:
    st.subheader("False-positive impact")
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric(
        "FP windows per 1,000 non-attack windows",
        f"{fp_impact['false_positive_windows_per_1000']:.2f}",
    )
    fcol1.caption(f"{fp_impact['false_positive_windows']} of {fp_impact['total_nonattack_windows']} held-out non-attack windows")
    fcol2.metric(
        "Legit txns affected per 1,000 non-attack txns",
        f"{fp_impact['estimated_legitimate_transactions_affected_per_1000']:.2f}",
    )
    fcol2.caption(
        f"{fp_impact['legitimate_transactions_in_false_positive_windows']} of "
        f"{fp_impact['total_legitimate_transactions_in_heldout_nonattack_windows']} legitimate held-out transactions"
    )
    fcol3.metric(
        "Legit transaction value temporarily affected",
        f"₹{fp_impact['estimated_legitimate_transaction_value_temporarily_affected']:,.0f}",
    )
    st.caption(
        "At the measured false-positive rate, some legitimate non-attack activity would be "
        "unnecessarily flagged for additional review."
    )
    st.caption(
        "These false positives represent customer friction and potentially delayed transaction "
        "value, not confirmed lost revenue."
    )
    st.caption(
        "Synthetic evaluation. Impact estimates are derived from held-out synthetic predictions "
        "and synthetic transaction amounts; they are not estimates of real merchant losses."
    )

counts = results["test_window_counts"]
st.caption(
    f"Primary-run evaluation: {results['attacks_detected']} holdout attacks detected; {counts['card_testing']} card-testing windows, "
    f"{counts['demand_spike']} demand-spike windows, and {counts['quiet_period']} quiet-period windows. "
    + notes.get("evaluation_scope", "Synthetic evaluation demonstrates the detection mechanism, not a production-scale false-positive rate.")
)
