 HEAD
# Fraud Spike Detector MVP

Synthetic, defense-only detection for card-testing fraud spikes. The prototype uses a two-stage pipeline: a frozen-baseline velocity detector identifies unusual traffic, then logistic regression separates card testing from legitimate demand spikes using decline rate, identity diversity, and amount features.

## Architecture

```text
Transaction simulator
        |
30-second rolling feature engine
        |
Stage 1: frozen-baseline velocity spike detection
        |
Stage 2: logistic-regression spike classification
        |
Explainable alerts + results.json + dashboard
        |
Evaluation against simulator ground truth
```

## Run

```bash
python fraud_spike_mvp.py
streamlit run dashboard.py
```

The first command generates `results.json` and `spike_detection_demo.png`; the dashboard only displays those already-computed artifacts.

## Evaluation

Card-testing detection: 12/12 holdout-test attacks detected with 0.00-minute median detection latency, and 0/528 non-attack evaluation windows (120 demand-spike, 408 quiet-period) triggered a false alert. This is a synthetic evaluation with 20 injected attacks across train/test, demonstrating that the detection mechanism works as designed — not a statistically robust production false-positive estimate. Production validation would require thousands of real merchant-days.

## Safety

Defense-only: this system detects and alerts only—it takes no offensive or evasive action and includes no code that could be repurposed for fraud.

