"""Single-file synthetic card-testing spike detection MVP.

Architecture (unchanged from the original prototype):

    Synthetic transaction generation
            |
    30-second transaction windows
            |
    Velocity spike detection (Stage 1)
            |
    Feature extraction
            |
    Logistic Regression classifier (Stage 2)
            |
    Card-testing probability
            |
    Confidence threshold
            |
    Fraud alerts
            |
    Evaluation metrics exported to results.json
            |
    Streamlit dashboard

What changed vs. the first version, and why, is explained in the
accompanying write-up. In short: every injected attack/demand-spike used to
be drawn from one fixed "template" per class (same decline probability, same
amount range, same fixed device/IP pool every time). That made the two
classes almost perfectly separable by construction, which is why precision/
recall/F1 all came out at 1.00. This version samples a *distribution* of
attacker/customer behaviours per injection (several named variants with
overlapping parameter ranges) so the classifier has to work with realistic
ambiguity, and it reports results averaged across several independently
generated scenarios (seeds) instead of a single run.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from statistics import mean, median, pstdev

# Avoid a Matplotlib cache warning in restricted/containerized environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fraud_spike_mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RNG_SEED = 42
# Every seed re-runs the full simulate -> train -> evaluate pipeline on a
# freshly generated synthetic scenario. RNG_SEED is always first and is the
# run used for the dashboard's detailed drill-down (velocity chart, alert
# list); the rest exist purely to show how much the headline metrics move
# across independent draws, instead of presenting one run as definitive.
AGGREGATE_SEEDS = [RNG_SEED, 101, 202, 303, 404, 505, 606, 707, 808, 909]

MERCHANT_ID = "merchant_001"
HOURS = 6
WINDOW_SECONDS = 30
SPIKE_SECONDS = 5 * 60
FEATURE_COLUMNS = [
    "decline_rate",
    "bin_diversity_ratio",
    "device_diversity_ratio",
    "ip_diversity_ratio",
    "pct_low_amount",
    "mean_amount",
]

# --------------------------------------------------------------------------
# Attack / demand-spike "variants".
#
# Each variant is a named behavioural pattern with a *range* for every
# parameter (not a fixed value), and each injection independently samples
# its own parameters from within that range. Ranges are chosen to overlap
# across card-testing and demand-spike variants on purpose (e.g. a
# "distributed" card-testing attack and a "mobile_app_event" demand spike
# can land on similar IP-diversity values) so the two classes are no longer
# trivially separable on any single feature.
# --------------------------------------------------------------------------

CARD_TESTING_VARIANTS = [
    {
        # The "obvious" pattern: few cards being tested for a few seconds,
        # heavy decline rate, from a handful of devices/IPs. Still the most
        # common attack shape, but no longer deterministic.
        "name": "classic_narrow",
        "weight": 0.45,
        "decline_prob": (0.60, 0.90),
        "amount_high": (30, 90),
        "device_pool_size": (3, 8),
        "ip_pool_size": (3, 8),
        "bin_pool_size": (5, 15),
    },
    {
        # A somewhat more careful attacker: moderate decline rate, a mix of
        # low and medium test amounts, slightly more devices/IPs.
        "name": "moderate_mix",
        "weight": 0.30,
        "decline_prob": (0.40, 0.68),
        "amount_high": (80, 180),
        "device_pool_size": (6, 18),
        "ip_pool_size": (6, 18),
        "bin_pool_size": (10, 30),
    },
    {
        # A distributed / botnet-style attacker spreading requests across
        # many devices and IPs to blend in with legitimate traffic.
        "name": "distributed",
        "weight": 0.25,
        "decline_prob": (0.28, 0.58),
        "amount_high": (60, 150),
        "device_pool_size": (20, 70),
        "ip_pool_size": (20, 70),
        "bin_pool_size": (20, 60),
    },
]

DEMAND_SPIKE_VARIANTS = [
    {
        # The "textbook" legitimate spike: broad diversity, high approval,
        # normal basket sizes.
        "name": "broad_clean",
        "weight": 0.40,
        "decline_prob": (0.02, 0.08),
        "amount_mean": (600, 1100),
        "amount_sigma": (0.8, 1.1),
        "device_pool_size": (150, 500),
        "ip_pool_size": (150, 500),
        "bin_pool_size": (80, 250),
    },
    {
        # A flash sale: many small purchases. Deliberately overlaps with
        # card-testing amount ranges above.
        "name": "flash_sale",
        "weight": 0.15,
        "decline_prob": (0.03, 0.10),
        "amount_mean": (40, 140),
        "amount_sigma": (0.6, 1.0),
        "device_pool_size": (150, 500),
        "ip_pool_size": (150, 500),
        "bin_pool_size": (60, 200),
    },
    {
        # A promo tied to a specific card partner: transactions concentrate
        # on a narrow BIN range even though everything else looks normal.
        "name": "promo_concentration",
        "weight": 0.15,
        "decline_prob": (0.03, 0.10),
        "amount_mean": (300, 900),
        "amount_sigma": (0.7, 1.1),
        "device_pool_size": (150, 500),
        "ip_pool_size": (150, 500),
        "bin_pool_size": (15, 45),
    },
    {
        # A temporary payment-gateway problem inflates the decline rate on
        # otherwise legitimate traffic.
        "name": "gateway_issue",
        "weight": 0.15,
        "decline_prob": (0.15, 0.35),
        "amount_mean": (400, 900),
        "amount_sigma": (0.8, 1.1),
        "device_pool_size": (150, 500),
        "ip_pool_size": (150, 500),
        "bin_pool_size": (80, 250),
    },
    {
        # A mobile-app promo: many distinct devices funnel through a small
        # number of carrier-grade NAT IPs, so IP diversity looks low even
        # though this is entirely legitimate demand.
        "name": "mobile_app_event",
        "weight": 0.15,
        "decline_prob": (0.03, 0.10),
        "amount_mean": (150, 500),
        "amount_sigma": (0.7, 1.1),
        "device_pool_size": (150, 500),
        "ip_pool_size": (8, 40),
        "bin_pool_size": (80, 250),
    },
]


def sample_weighted(rng: np.random.Generator, options: list[dict]) -> dict:
    weights = np.array([option["weight"] for option in options], dtype=float)
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


def sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def attach_injection_profile(
    rng: np.random.Generator, injection: dict, all_bins: np.ndarray, devices: np.ndarray, ips: np.ndarray
) -> dict:
    """Sample a behavioural variant + its parameters for one injection window.

    Each injection gets its own device/IP/BIN pools (drawn fresh from the
    full universe) instead of every attack sharing one fixed pool, and its
    own decline probability / amount profile sampled from a range rather
    than a constant. This is what removes the "every attack looks
    identical" artifact from the original simulator.
    """
    variants = CARD_TESTING_VARIANTS if injection["label"] == "card_testing" else DEMAND_SPIKE_VARIANTS
    variant = sample_weighted(rng, variants)
    injection["variant"] = variant["name"]

    bin_low, bin_high = variant["bin_pool_size"]
    bin_pool_size = int(rng.integers(bin_low, min(bin_high, len(all_bins)) + 1))
    injection["bin_pool_size"] = bin_pool_size
    injection["bin_pool"] = rng.choice(all_bins, size=bin_pool_size, replace=False)

    dev_low, dev_high = variant["device_pool_size"]
    device_pool_size = int(rng.integers(dev_low, min(dev_high, len(devices)) + 1))
    injection["device_pool"] = rng.choice(devices, size=device_pool_size, replace=False)

    ip_low, ip_high = variant["ip_pool_size"]
    ip_pool_size = int(rng.integers(ip_low, min(ip_high, len(ips)) + 1))
    injection["ip_pool"] = rng.choice(ips, size=ip_pool_size, replace=False)

    injection["decline_prob"] = sample_range(rng, variant["decline_prob"])
    if injection["label"] == "card_testing":
        injection["amount_low"] = 1.0
        injection["amount_high"] = sample_range(rng, variant["amount_high"])
    else:
        injection["amount_mean"] = sample_range(rng, variant["amount_mean"])
        injection["amount_sigma"] = sample_range(rng, variant["amount_sigma"])
    return injection


def choose_injection_windows(
    rng: np.random.Generator, start: pd.Timestamp, merchant_id: str, attack_count: int, demand_count: int
) -> list[dict]:
    """Choose randomized, non-overlapping five-minute windows across train/test."""
    # Keep examples of both classes in train for fitting and in holdout for
    # evaluation. Five-minute slots make non-overlap explicit and reproducible.
    train_attack_count = max(1, round(attack_count * 0.4))
    train_demand_count = max(1, round(demand_count * 0.4))
    labels = (["card_testing"] * train_attack_count + ["demand_spike"] * train_demand_count
              + ["card_testing"] * (attack_count - train_attack_count)
              + ["demand_spike"] * (demand_count - train_demand_count))
    train_slots = rng.choice(np.arange(0, 250, 5), size=train_attack_count + train_demand_count, replace=False)
    test_slots = rng.choice(np.arange(255, 360, 5), size=len(labels) - len(train_slots), replace=False)
    rng.shuffle(train_slots)
    rng.shuffle(test_slots)
    selected: list[dict] = []
    for label, minute in zip(labels, np.concatenate([train_slots, test_slots])):
        selected.append({
            "merchant_id": merchant_id,
            "label": label,
            "start": start + pd.Timedelta(minutes=int(minute)),
            "end": start + pd.Timedelta(minutes=int(minute) + 5),
        })
    return sorted(selected, key=lambda item: item["start"])


def lognormal_amount(rng: np.random.Generator, target_mean: float, sigma: float, count: int) -> np.ndarray:
    """numpy takes log-space mean; convert from requested monetary mean."""
    log_mu = np.log(target_mean) - (sigma**2) / 2
    return rng.lognormal(mean=log_mu, sigma=sigma, size=count)


def simulate_merchant(
    merchant_id: str, attack_count: int, demand_count: int, rng: np.random.Generator, start: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict]]:
    end = start + pd.Timedelta(hours=HOURS)
    all_bins = np.array([f"{400000 + n:06d}" for n in range(250)])
    bins = all_bins[:200]  # Normal traffic retains the requested 200-BIN diversity.
    devices = np.array([f"device_{n:03d}" for n in range(500)])
    ips = np.array([f"10.0.{n // 250}.{n % 250 + 1}" for n in range(500)])
    injections = choose_injection_windows(rng, start, merchant_id, attack_count, demand_count)
    for injection in injections:
        attach_injection_profile(rng, injection, all_bins, devices, ips)

    rows: list[dict] = []
    intervals = pd.date_range(start, end, freq="10s", inclusive="left")
    for interval_start in intervals:
        active = next((x for x in injections if x["start"] <= interval_start < x["end"]), None)
        scenario = active["label"] if active else "normal"
        count = rng.poisson(10 if active else 1)  # 1 txn / 10 sec, 10x during injections
        if count == 0:
            continue
        timestamps = interval_start + pd.to_timedelta(rng.uniform(0, 10, count), unit="s")
        if scenario == "card_testing":
            base_amounts = rng.uniform(active["amount_low"], active["amount_high"], count)
            # A small fraction of card-testing traffic includes a larger
            # "verification" charge alongside the usual micro-charges, which
            # pulls some attack amounts up into legitimate territory.
            is_probe = rng.random(count) < 0.08
            amounts = np.where(is_probe, base_amounts * rng.uniform(1.5, 3.0, count), base_amounts)
            row_bins = rng.choice(active["bin_pool"], count)
            row_devices = rng.choice(active["device_pool"], count)
            row_ips = rng.choice(active["ip_pool"], count)
            approved = rng.random(count) >= active["decline_prob"]
        elif scenario == "demand_spike":
            amounts = lognormal_amount(rng, active["amount_mean"], active["amount_sigma"], count)
            row_bins = rng.choice(active["bin_pool"], count)
            row_devices = rng.choice(active["device_pool"], count)
            row_ips = rng.choice(active["ip_pool"], count)
            approved = rng.random(count) >= active["decline_prob"]
        else:
            amounts = lognormal_amount(rng, 500, 1, count)
            row_bins, row_devices, row_ips = rng.choice(bins, count), rng.choice(devices, count), rng.choice(ips, count)
            approved = rng.random(count) < 0.95
        for i in range(count):
            rows.append({"timestamp": timestamps[i], "merchant_id": merchant_id, "amount": amounts[i],
                         "card_bin": row_bins[i], "device_id": row_devices[i], "ip": row_ips[i],
                         "approved": bool(approved[i]), "scenario_label": scenario})
    transactions = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return transactions, injections


def simulate_transactions(seed: int) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(datetime.now().replace(microsecond=0, second=0, minute=0))
    merchant_specs = [("merchant_001", 10, 10), ("merchant_002", 5, 5), ("merchant_003", 5, 5)]
    transaction_frames, all_injections = [], []
    for merchant_id, attacks, demand_spikes in merchant_specs:
        merchant_transactions, injections = simulate_merchant(merchant_id, attacks, demand_spikes, rng, start)
        transaction_frames.append(merchant_transactions)
        all_injections.extend(injections)
    return pd.concat(transaction_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True), all_injections


def majority_label(values: pd.Series) -> str:
    counted = Counter(values)
    return counted.most_common(1)[0][0] if counted else np.nan


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    feature_frames = []
    for merchant_id, merchant_transactions in transactions.groupby("merchant_id", sort=True):
        indexed = merchant_transactions.set_index("timestamp")
        grouped = indexed.resample(f"{WINDOW_SECONDS}s")
        windows = grouped.agg(
            txn_count=("amount", "size"),
            decline_rate=("approved", lambda s: 1 - s.mean()),
            mean_amount=("amount", "mean"),
            unique_bins=("card_bin", "nunique"),
            unique_devices=("device_id", "nunique"),
            unique_ips=("ip", "nunique"),
            pct_low_amount=("amount", lambda s: (s < 50).mean()),
            majority_scenario_label=("scenario_label", majority_label),
        ).dropna(subset=["txn_count"])
        windows["merchant_id"] = merchant_id
        windows["velocity"] = windows["txn_count"] / WINDOW_SECONDS
        for source, target in [("unique_bins", "bin_diversity_ratio"),
                               ("unique_devices", "device_diversity_ratio"),
                               ("unique_ips", "ip_diversity_ratio")]:
            windows[target] = windows[source] / windows["txn_count"]
        # Freeze the per-merchant normal baseline during a candidate spike.
        # This loop only ever looks at *past* windows to decide whether the
        # current one is a spike, so Stage 1 cannot peek at future data.
        normal_baseline: list[float] = []
        rolling_means, rolling_stds, candidates = [], [], []
        for velocity in windows["velocity"]:
            if len(normal_baseline) < 10:
                rolling_means.append(np.nan)
                rolling_stds.append(np.nan)
                is_spike = False
            else:
                baseline_mean = float(np.mean(normal_baseline))
                baseline_std = float(np.std(normal_baseline, ddof=1))
                rolling_means.append(baseline_mean)
                rolling_stds.append(baseline_std)
                is_spike = velocity > baseline_mean + 3 * baseline_std
            candidates.append(is_spike)
            if not is_spike:
                normal_baseline.append(float(velocity))
                if len(normal_baseline) > 10:
                    normal_baseline.pop(0)
        windows["velocity_rolling_mean"] = rolling_means
        windows["velocity_rolling_std"] = rolling_stds
        windows["candidate_spike"] = candidates
        windows.index.name = "window_start"
        feature_frames.append(windows.reset_index())
    return pd.concat(feature_frames, ignore_index=True).sort_values(["window_start", "merchant_id"]).reset_index(drop=True)


def compute_legitimate_window_stats(transactions: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Bucket every non-card-testing ("legitimate") transaction into the same
    fixed-width, per-merchant window that build_features uses, and aggregate
    the actual transaction count/amount per (merchant_id, window_start).

    30 seconds divides evenly into a day, so flooring each timestamp to the
    window width lands on the same bin edges as build_features' per-merchant
    resample(f"{WINDOW_SECONDS}s") (pandas' default resample origin is
    start-of-day, which is itself a multiple of the window width). This lets
    us join real synthetic transaction data onto the evaluation's window-level
    predictions without re-deriving or estimating anything.
    """
    legit = transactions[transactions["scenario_label"] != "card_testing"].copy()
    if legit.empty:
        return pd.DataFrame(columns=["merchant_id", "window_start", "legit_txn_count", "legit_txn_amount"])
    legit["window_start"] = legit["timestamp"].dt.floor(f"{window_seconds}s")
    return (
        legit.groupby(["merchant_id", "window_start"])
        .agg(legit_txn_count=("amount", "size"), legit_txn_amount=("amount", "sum"))
        .reset_index()
    )


def compute_false_positive_impact(
    test: pd.DataFrame, transactions: pd.DataFrame, truth: pd.Series, predicted: np.ndarray, window_seconds: int
) -> dict:
    """Honest, dynamically computed false-positive *impact* metrics for the
    held-out test split — window-level false-positive counts plus the actual
    legitimate transactions/value inside those falsely flagged windows.

    A false positive here means a held-out window that was NOT a card-testing
    attack (i.e. a legitimate demand-spike window or a quiet-period window)
    but was alerted on anyway. This does not claim the underlying transaction
    value was lost — only that it sat inside a window that would have been
    flagged for review. All values are derived from the actual held-out
    predictions and the actual simulated transactions; nothing is estimated
    from an average or a hardcoded constant, and every ratio is guarded
    against division by zero.
    """
    non_attack = ~truth
    fp_mask = predicted & non_attack

    false_positive_windows = int(fp_mask.sum())
    total_nonattack_windows = int(non_attack.sum())
    false_positive_windows_per_1000 = (
        false_positive_windows / total_nonattack_windows * 1000 if total_nonattack_windows > 0 else 0.0
    )

    legit_window_stats = compute_legitimate_window_stats(transactions, window_seconds)
    test_with_legit = test.merge(legit_window_stats, on=["merchant_id", "window_start"], how="left")
    test_with_legit[["legit_txn_count", "legit_txn_amount"]] = (
        test_with_legit[["legit_txn_count", "legit_txn_amount"]].fillna(0.0)
    )
    # merge preserves row order/index alignment with `test`, so the same
    # boolean masks (fp_mask, non_attack) computed against `test` apply here.
    fp_mask_arr = fp_mask.to_numpy() if hasattr(fp_mask, "to_numpy") else np.asarray(fp_mask)
    non_attack_arr = non_attack.to_numpy() if hasattr(non_attack, "to_numpy") else np.asarray(non_attack)

    legitimate_transactions_in_false_positive_windows = int(
        test_with_legit.loc[fp_mask_arr, "legit_txn_count"].sum()
    )
    total_legitimate_transactions_in_heldout_nonattack_windows = int(
        test_with_legit.loc[non_attack_arr, "legit_txn_count"].sum()
    )
    estimated_legitimate_transactions_affected_per_1000 = (
        legitimate_transactions_in_false_positive_windows
        / total_legitimate_transactions_in_heldout_nonattack_windows
        * 1000
        if total_legitimate_transactions_in_heldout_nonattack_windows > 0
        else 0.0
    )
    estimated_legitimate_transaction_value_temporarily_affected = float(
        test_with_legit.loc[fp_mask_arr, "legit_txn_amount"].sum()
    )

    return {
        "false_positive_windows": false_positive_windows,
        "total_nonattack_windows": total_nonattack_windows,
        "false_positive_windows_per_1000": false_positive_windows_per_1000,
        "legitimate_transactions_in_false_positive_windows": legitimate_transactions_in_false_positive_windows,
        "total_legitimate_transactions_in_heldout_nonattack_windows": (
            total_legitimate_transactions_in_heldout_nonattack_windows
        ),
        "estimated_legitimate_transactions_affected_per_1000": estimated_legitimate_transactions_affected_per_1000,
        "estimated_legitimate_transaction_value_temporarily_affected": (
            estimated_legitimate_transaction_value_temporarily_affected
        ),
        "currency": "INR",
        "interpretation": (
            "False positives represent customer friction and potentially delayed transaction "
            "value, not confirmed lost revenue."
        ),
    }


def format_latency_display(latency_minutes: float | None, window_seconds: int) -> str:
    """Render a measured latency (in minutes, unchanged from the existing
    calculation) as an honest, appropriately-scaled string instead of a
    fixed-precision minute value. Detection is only ever observed at
    WINDOW_SECONDS resolution, so a sub-minute latency is shown in seconds,
    and an exact-zero latency (attack caught in its first observable window)
    is reported as such rather than implied to be some small nonzero delay.
    """
    if latency_minutes is None:
        return "no alert"
    seconds = latency_minutes * 60
    if seconds <= 0:
        return "0 sec"
    if seconds < 60:
        rounded_seconds = round(seconds)
        return f"{rounded_seconds} sec" if rounded_seconds > 0 else "<1 sec"
    return f"{latency_minutes:.2f} min"


def run_pipeline(seed: int, verbose: bool) -> dict:
    """Simulate one scenario end-to-end and evaluate it. Returns everything
    needed both for the aggregate metrics table and (for the primary seed)
    the dashboard's detailed drill-down."""
    if verbose:
        print(f"=== Fraud Spike MVP: generating synthetic merchant transactions (seed={seed}) ===")
    transactions, injections = simulate_transactions(seed)
    if verbose:
        print(f"Generated {len(transactions):,} transactions across {transactions.merchant_id.nunique()} merchants over {HOURS} hours.")
        variant_counts = pd.DataFrame(injections).groupby(["label", "variant"]).size()
        print("Injected scenario variants:")
        print(variant_counts.to_string())

    features = build_features(transactions)
    split_time = features["window_start"].min() + pd.Timedelta(hours=HOURS * 0.70)
    train, test = features[features["window_start"] < split_time].copy(), features[features["window_start"] >= split_time].copy()
    if verbose:
        print("\n=== Rolling 30-second feature windows ===")
        print(f"Created {len(features)} windows: {len(train)} train / {len(test)} test.")
        print("Candidate spikes by split:", {"train": int(train.candidate_spike.sum()), "test": int(test.candidate_spike.sum())})

    train_candidates = train[train["candidate_spike"] & train["majority_scenario_label"].isin(["card_testing", "demand_spike"])].copy()
    if verbose:
        print("\n=== Stage 2 training ===")
        print(f"Eligible train candidate windows: {len(train_candidates)}; labels: {train_candidates.majority_scenario_label.value_counts().to_dict()}")
    if train_candidates["majority_scenario_label"].nunique() < 2:
        raise RuntimeError(f"seed {seed}: training data lacks both injected spike classes")

    # StandardScaler + LogisticRegression are fit ONLY on train_candidates;
    # the fitted scaler's mean/std and the model's coefficients never see
    # any test-split row, so there is no train/test contamination here.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(train_candidates[FEATURE_COLUMNS], train_candidates["majority_scenario_label"])
    if verbose:
        print("Trained LogisticRegression on:", ", ".join(FEATURE_COLUMNS))

    test["predicted_label"] = "normal"
    test["card_testing_probability"] = 0.0
    test_candidates = test[test["candidate_spike"]].copy()
    if not test_candidates.empty:
        probabilities = model.predict_proba(test_candidates[FEATURE_COLUMNS])
        card_index = list(model.classes_).index("card_testing")
        test.loc[test_candidates.index, "card_testing_probability"] = probabilities[:, card_index]
        test.loc[test_candidates.index, "predicted_label"] = model.predict(test_candidates[FEATURE_COLUMNS])
        if verbose:
            print("Test candidate card-testing probability range: "
                  f"min={probabilities[:, card_index].min():.3f}, max={probabilities[:, card_index].max():.3f}")
    # Alert threshold (0.60) is a fixed a-priori decision rule, not tuned
    # against these test-split predictions.
    alerts = test[(test["predicted_label"] == "card_testing") & (test["card_testing_probability"] > 0.60)].copy()

    if verbose:
        print("\n=== Alerts fired (test candidate windows only) ===")
        if alerts.empty:
            print("No card-testing alerts met the > 0.60 confidence threshold.")
        for _, alert in alerts.iterrows():
            detail = ", ".join(f"{feature}={alert[feature]:.3f}" for feature in FEATURE_COLUMNS)
            print(f"ALERT merchant={alert.merchant_id} window={alert.window_start} to {alert.window_start + pd.Timedelta(seconds=WINDOW_SECONDS)} "
                  f"type=card_testing model_score={alert.card_testing_probability:.3f} | {detail}")

    truth = test["majority_scenario_label"].eq("card_testing")
    predicted = test.index.isin(alerts.index)
    precision = precision_score(truth, predicted, zero_division=0)
    recall = recall_score(truth, predicted, zero_division=0)
    f1 = f1_score(truth, predicted, zero_division=0)
    demand = test["majority_scenario_label"].eq("demand_spike")
    fp_rate = float(predicted[demand].mean()) if demand.any() else float("nan")
    non_attack = ~truth
    fp_all_rate = float(predicted[non_attack].mean()) if non_attack.any() else float("nan")
    false_positive_impact = compute_false_positive_impact(test, transactions, truth, predicted, WINDOW_SECONDS)

    if verbose:
        print("\n=== Full test-set evaluation ===")
        print(f"Card-testing precision: {precision:.3f}")
        print(f"Card-testing recall:    {recall:.3f}")
        print(f"Card-testing F1:        {f1:.3f}")
        print(f"False-positive rate on demand_spike windows: {fp_rate:.3f}")
        print(f"False-positive rate on all non-attack windows: {fp_all_rate:.3f}")
        print("\n=== False-positive impact (held-out) ===")
        print(f"False-positive windows: {false_positive_impact['false_positive_windows']}/"
              f"{false_positive_impact['total_nonattack_windows']} non-attack windows "
              f"({false_positive_impact['false_positive_windows_per_1000']:.2f} per 1,000)")
        print(f"Legitimate transactions in false-positive windows: "
              f"{false_positive_impact['legitimate_transactions_in_false_positive_windows']}/"
              f"{false_positive_impact['total_legitimate_transactions_in_heldout_nonattack_windows']} "
              f"({false_positive_impact['estimated_legitimate_transactions_affected_per_1000']:.2f} per 1,000)")
        print("Estimated legitimate transaction value temporarily affected: "
              f"₹{false_positive_impact['estimated_legitimate_transaction_value_temporarily_affected']:,.2f} "
              "(not confirmed revenue lost)")
        print(f"Test-set cardinality: {int(truth.sum())} card-testing windows; {int(demand.sum())} demand-spike windows; "
              f"{int((~truth & ~demand).sum())} quiet-period windows")

    attack_level_results = []
    for attack in (item for item in injections if item["label"] == "card_testing"):
        matching = alerts[(alerts["merchant_id"] == attack["merchant_id"])
                          & (alerts["window_start"] >= attack["start"])
                          & (alerts["window_start"] < attack["end"])]
        if matching.empty:
            attack_level_results.append({"merchant_id": attack["merchant_id"], "attack_start": attack["start"],
                                          "variant": attack["variant"], "detected": False, "latency_min": None})
        else:
            latency = (matching["window_start"].iloc[0] - attack["start"]).total_seconds() / 60
            attack_level_results.append({"merchant_id": attack["merchant_id"], "attack_start": attack["start"],
                                          "variant": attack["variant"], "detected": True, "latency_min": latency})
    if verbose:
        print("\nDetection latency for injected card-testing attacks:")
        for item in attack_level_results:
            scope = format_latency_display(item["latency_min"], WINDOW_SECONDS) if item["detected"] else "no alert"
            print(f"  {item['merchant_id']} {item['attack_start']} [{item['variant']}]: {scope}")

    test_attacks = [item for item in attack_level_results if item["attack_start"] >= test["window_start"].min()]
    test_detected = sum(item["detected"] for item in test_attacks)
    detected_latencies = [item["latency_min"] for item in attack_level_results if item["detected"]]
    median_latency = float(median(detected_latencies)) if detected_latencies else None
    if verbose:
        print(f"Attack-level detection (test alerts): {sum(1 for i in attack_level_results if i['detected'])}/{len(attack_level_results)} attacks detected")
        print(f"Holdout attack coverage: {test_detected}/{len(test_attacks)} test-set attacks detected (small sample)")
        print(f"Median detection latency: {format_latency_display(median_latency, WINDOW_SECONDS)}"
              if median_latency is not None else "Median detection latency: no detected attacks")

    return {
        "seed": seed,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_rate_demand_spike": fp_rate,
        "fp_rate_all_nonattack": fp_all_rate,
        "test_window_counts": {
            "card_testing": int(truth.sum()),
            "demand_spike": int(demand.sum()),
            "quiet_period": int((~truth & ~demand).sum()),
        },
        "attacks_detected": f"{test_detected}/{len(test_attacks)}",
        "median_latency_min": median_latency,
        "median_latency_display": format_latency_display(median_latency, WINDOW_SECONDS),
        "false_positive_impact": false_positive_impact,
        "features": features,
        "injections": injections,
        "alerts": alerts,
        "test": test,
    }


def summarize_seeds(results: list[dict]) -> dict:
    """Aggregate precision/recall/F1/FP-rate across independently generated
    scenarios so a single lucky (or unlucky) draw isn't presented as THE
    result."""
    def agg(key: str) -> dict:
        values = [r[key] for r in results]
        return {"mean": float(mean(values)), "std": float(pstdev(values)) if len(values) > 1 else 0.0}

    return {
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "precision": agg("precision"),
        "recall": agg("recall"),
        "f1": agg("f1"),
        "fp_rate_all_nonattack": agg("fp_rate_all_nonattack"),
        "fp_rate_demand_spike": agg("fp_rate_demand_spike"),
        "per_seed": [
            {"seed": r["seed"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
             "fp_rate_all_nonattack": r["fp_rate_all_nonattack"]}
            for r in results
        ],
    }


def main() -> None:
    seed_results = []
    primary = None
    for seed in AGGREGATE_SEEDS:
        try:
            result = run_pipeline(seed, verbose=(seed == RNG_SEED))
        except RuntimeError as exc:
            print(f"Seed {seed}: skipped ({exc})")
            continue
        seed_results.append(result)
        if seed == RNG_SEED:
            primary = result
    if primary is None:
        raise RuntimeError("Primary seed produced no usable result; cannot build dashboard output.")

    aggregate = summarize_seeds(seed_results)
    print(f"\n=== Aggregate evaluation across {aggregate['n_seeds']} independent synthetic scenarios ===")
    print(f"Precision: mean={aggregate['precision']['mean']:.3f} std={aggregate['precision']['std']:.3f}")
    print(f"Recall:    mean={aggregate['recall']['mean']:.3f} std={aggregate['recall']['std']:.3f}")
    print(f"F1:        mean={aggregate['f1']['mean']:.3f} std={aggregate['f1']['std']:.3f}")
    print(f"FP rate (all non-attack): mean={aggregate['fp_rate_all_nonattack']['mean']:.3f} "
          f"std={aggregate['fp_rate_all_nonattack']['std']:.3f}")
    print("Per-seed precision/recall/F1:")
    for row in aggregate["per_seed"]:
        print(f"  seed={row['seed']:>4}  precision={row['precision']:.3f}  recall={row['recall']:.3f}  "
              f"f1={row['f1']:.3f}  fp_all={row['fp_rate_all_nonattack']:.3f}")

    features, injections, alerts, test = primary["features"], primary["injections"], primary["alerts"], primary["test"]
    alert_feature_cols = FEATURE_COLUMNS
    results = {
        "primary_seed": RNG_SEED,
        "precision": primary["precision"],
        "recall": primary["recall"],
        "f1": primary["f1"],
        "attacks_detected": primary["attacks_detected"],
        "median_latency_min": primary["median_latency_min"],
        "median_latency_display": primary["median_latency_display"],
        "window_seconds": WINDOW_SECONDS,
        "fp_rate_demand_spike": primary["fp_rate_demand_spike"],
        "fp_rate_all_nonattack": primary["fp_rate_all_nonattack"],
        "test_window_counts": primary["test_window_counts"],
        "false_positive_impact": primary["false_positive_impact"],
        "aggregate_evaluation": aggregate,
        "methodology_notes": {
            "score_interpretation": (
                "Alert 'model_score' is the logistic regression's own predicted probability for the "
                "card_testing class on held-out synthetic windows. It is a relative ranking signal, not "
                "a calibrated real-world probability of fraud."
            ),
            "evaluation_scope": (
                f"All metrics are computed on held-out synthetic scenarios ({aggregate['n_seeds']} independently "
                "generated seeds), not on production traffic. They demonstrate the two-stage detection "
                "mechanism, not a production-scale false-positive rate."
            ),
            "no_leakage_found": (
                "StandardScaler and LogisticRegression are fit only on train-split candidate windows; the "
                "0.60 alert threshold is fixed a priori and not tuned on test data; Stage 1's rolling "
                "baseline only looks backward in time. No train/test contamination was found in the "
                "current pipeline."
            ),
            "latency_interpretation": (
                f"Detection latency is measured at the simulation/window resolution ({WINDOW_SECONDS}-second "
                "windows) and is not a production network-latency benchmark. A latency of 0 sec means the "
                "attack was detected within the same window it started, not that detection was instantaneous."
            ),
        },
        "alerts": [
            {
                "merchant_id": alert.merchant_id,
                "window_start": str(alert.window_start),
                "window_end": str(alert.window_start + pd.Timedelta(seconds=WINDOW_SECONDS)),
                "model_score": float(alert.card_testing_probability),
                "spike_type": "card_testing",
                "features": {col: float(getattr(alert, col)) for col in alert_feature_cols},
            }
            for _, alert in alerts.iterrows()
        ],
        "velocity_series": [
            {"merchant_id": row.merchant_id, "window_start": str(row.window_start), "velocity": float(row.velocity)}
            for _, row in features.iterrows()
        ],
        "injections": [
            {"merchant_id": item["merchant_id"], "label": item["label"], "variant": item["variant"],
             "start": str(item["start"]), "end": str(item["end"])}
            for item in injections
        ],
    }
    with open("results.json", "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
    print("\nResults exported to results.json")

    print("\n=== Saving plot ===")
    display_merchant = "merchant_001"
    display_features = features[features["merchant_id"] == display_merchant]
    display_alerts = alerts[alerts["merchant_id"] == display_merchant]
    plt.figure(figsize=(14, 6))
    plt.plot(display_features["window_start"], display_features["velocity"], color="steelblue", linewidth=1.4, label="Velocity (txns/sec)")
    for i, item in enumerate(item for item in injections if item["merchant_id"] == display_merchant):
        color = "red" if item["label"] == "card_testing" else "green"
        name = "True attack start" if item["label"] == "card_testing" else "True demand-spike start"
        plt.axvline(item["start"], color=color, linestyle="--", alpha=0.75, label=name if i < 2 else None)
    if not display_alerts.empty:
        plt.scatter(display_alerts["window_start"], display_alerts["velocity"], marker="^", s=70, color="black", label="Alert fired", zorder=3)
    plt.title(f"{display_merchant} velocity: injected spikes and fired alerts (seed={RNG_SEED})")
    plt.xlabel("Window start")
    plt.ylabel("Transactions per second")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spike_detection_demo.png", dpi=150)
    print("Saved spike_detection_demo.png")


if __name__ == "__main__":
    main()
