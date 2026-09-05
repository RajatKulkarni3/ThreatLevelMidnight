# ThreatLevelMidnight — Real-Time Card-Testing Spike Detector

**Track 2: AI Risk Manager — Razorpay AI Buildathon**

## The problem

Confirmed-fraud (chargeback) labels take 30-120 days to arrive. A classifier trained
on confirmed labels is structurally blind to an attack happening *right now* — by the
time a label exists, the damage is done.

This project detects that a merchant is **currently under a coordinated card-testing
attack** — from behavioral pattern alone, before any confirmed label exists — and
distinguishes it from a **legitimate demand spike** (e.g. a flash sale), since both
look identical at a glance: a sudden burst of transaction volume.

- **Card-testing attack**: stolen card numbers run through checkout in small,
  low-value charges to find which ones still work. Signature: burst of volume, high
  decline rate, narrow BIN pool, narrow device/IP pool, low amounts.
- **Legitimate demand spike**: also a sudden volume burst, but high approval rate,
  wide device/card/IP diversity, normal-to-higher amounts.

## Architecture

```
Synthetic transaction simulator (labeled: card_testing / demand_spike / normal)
        |
Rolling 30-second feature windows, per merchant
  (velocity, decline_rate, bin/device/ip diversity ratios, pct_low_amount, mean_amount)
        |
Stage 1 — change-point detector
  rolling baseline mean/std on velocity; a window is a candidate spike when
  velocity > baseline_mean + 3 x baseline_std.
  The baseline FREEZES while a spike is active, so a sustained attack doesn't get
  absorbed into "normal" and stop triggering.
        |
Stage 2 — classifier (only runs on candidate-spike windows)
  StandardScaler -> LogisticRegression, predicting card_testing vs demand_spike
        |
Alert engine
  fires when predicted_label == card_testing AND model_score > 0.60
  (model_score is the classifier's own probability estimate — an uncalibrated
  ranking signal, not a calibrated real-world fraud probability)
        |
Evaluation harness
  window-level precision/recall/F1, false-positive rate against BOTH demand-spike
  windows and ALL non-attack windows (including quiet periods), plus attack-level
  detection (did we catch each true injected attack at all, and how fast) —
  reported across 10 independently generated synthetic scenarios, not one run.
        |
Streamlit dashboard (dashboard.py)
  velocity chart with true attack/demand starts and fired alerts, per-alert feature
  breakdown, aggregate metrics across scenarios.
```

## How to run

```bash
pip install -r requirements.txt
python3 fraud_spike_mvp.py            # generates results.json + spike_detection_demo.png
python3 -m streamlit run dashboard.py # interactive dashboard
```

## Evaluation results (real, not hardcoded)

**Primary run (seed 42)** — 3 merchants, 6 hours of synthetic traffic, 20 injected
card-testing attacks and 20 injected demand spikes total, 648 held-out test windows
(120 card-testing, 120 demand-spike, 408 quiet):

| Metric | Value |
|---|---|
| Precision | 0.937 |
| Recall | 0.992 |
| F1 | 0.964 |
| False-positive rate (demand-spike windows) | 6.7% |
| False-positive rate (all non-attack windows) | 1.5% |
| Attack-level detection | 12/12 test-split attacks caught |
| Median detection latency | 0 sec (measured at 30-second window resolution — detected within the same window the attack started; not a production network-latency benchmark) |

**Aggregate across 10 independently generated synthetic scenarios** (fresh data each
time, same pipeline, seeds 42/101/202/303/404/505/606/707/808/909):

| Metric | Mean | Std dev | Range across seeds |
|---|---|---|---|
| Precision | 0.993 | 0.019 | 0.937 – 1.000 |
| Recall | 0.953 | 0.082 | 0.758 – 1.000 |
| F1 | 0.970 | 0.046 | 0.863 – 1.000 |
| FP rate (all non-attack) | 0.2% | 0.5% | 0.0% – 1.5% |

We report the spread deliberately: a single run's numbers are not presented as *the*
result. Recall is the metric that moves the most (down to 0.76 on one seed) — that's
the honest signal that the underlying synthetic attacks and demand spikes now overlap
in behavior rather than being trivially separable.

## False-positive impact

A false positive does not necessarily mean revenue is permanently lost. ThreatLevelMidnight
is a defense-only detection system: an alert represents additional review or temporary
friction rather than an automatic permanent rejection.

Therefore, false-positive impact is reported operationally using metrics derived directly
from the held-out evaluation (primary run, seed 42):

| Metric | Value |
|---|---|
| False-positive windows per 1,000 non-attack windows | 15.15 (8 / 528) |
| Estimated legitimate transactions affected per 1,000 non-attack transactions | 51.32 (245 / 4,774) |
| Estimated legitimate transaction value temporarily affected | ₹10,213.06 |

The transaction-value figure is not reported as revenue lost. It represents transaction
value associated with legitimate activity that may experience unnecessary review or delay.
It is computed by summing the actual synthetic transaction amounts belonging to held-out
windows that were not card-testing attacks but were alerted on anyway — no averaging or
hardcoded per-transaction cost is used, and a scenario with zero false positives reports
zero for every field in this section rather than a placeholder value.

> Synthetic evaluation. Impact estimates are derived from measured held-out predictions and
> the synthetic transaction distribution; they are not estimates of real merchant losses.

## Known limitations

- **Synthetic data.** Real card-testing attack windows are proprietary to payment
  processors and don't exist in any public dataset, so this cannot be validated
  against real attack traffic. Transaction amount and decline-rate baselines are
  loosely calibrated against public dataset statistics (e.g. the Kaggle Credit Card
  Fraud dataset) rather than invented from nothing, but the labeled attack/demand
  patterns themselves are simulated.
- **Cold-start blind spot.** Stage 1 needs ~10 quiet windows (5 minutes) of baseline
  before it can flag anything as a spike. An attack occurring in that warm-up period
  will not be caught until the baseline exists. This is a structural property of the
  design, not random noise, and accounts for some of the misses in the results above.
- **Recall varies meaningfully across scenarios** (0.76-1.00) — this is not a
  reliably "solved" detector; it is a working prototype whose failure modes are
  measured and disclosed rather than hidden.
- **Model score is uncalibrated.** The classifier's probability output is a relative
  ranking signal for thresholding alerts, not a calibrated real-world likelihood of
  fraud.
- **Single alert threshold, not adaptive.** The 0.60 cutoff is fixed and was not
  tuned against held-out data, but it also has not been validated across a wider
  range of merchant traffic patterns than the three simulated here.
- **Simplified Stage 1.** A production system would likely use a proper streaming
  change-point method (EWMA/CUSUM) rather than a fixed rolling z-score threshold.

## Defense-only statement

This system is strictly detection and recommendation. It emits alerts with a
suggested bounded action (e.g. "escalate the flagged activity for additional review;
any payment intervention should remain subject to merchant risk policy and human
approval") for a human or a separate authorization system to act on. It does not
autonomously block, cancel, or reverse any transaction, and it has no
offense-capable functionality of any kind.
