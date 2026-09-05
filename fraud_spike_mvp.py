"""Single-file synthetic card-testing spike detection MVP."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json
import os
from statistics import median

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
            "bin_pool_size": int(rng.integers(5, 21)) if label == "card_testing" else None,
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
    narrow_devices, narrow_ips = devices[:5], ips[:5]
    injections = choose_injection_windows(rng, start, merchant_id, attack_count, demand_count)
    for injection in injections:
        if injection["label"] == "card_testing":
            injection["bin_pool"] = rng.choice(all_bins, size=injection["bin_pool_size"], replace=False)
        else:
            # Most demand spikes retain high diversity; a small fraction model a
            # legitimate popular-SKU spike with moderately concentrated BINs.
            narrow_demand = rng.random() < 0.10
            low, high = (40, 60) if narrow_demand else (80, len(all_bins))
            pool_size = int(rng.integers(low, high + 1))
            injection["bin_pool_size"] = pool_size
            injection["bin_pool"] = rng.choice(all_bins, size=pool_size, replace=False)

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
            amounts = rng.uniform(1, 60, count)
            row_bins = rng.choice(active["bin_pool"], count)
            row_devices = rng.choice(narrow_devices, count)
            row_ips = rng.choice(narrow_ips, count)
            approved = rng.random(count) >= 0.80
        elif scenario == "demand_spike":
            amounts = lognormal_amount(rng, 800, 1, count)
            row_bins = rng.choice(active["bin_pool"], count)
            row_devices, row_ips = rng.choice(devices, count), rng.choice(ips, count)
            approved = rng.random(count) < 0.95
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


def simulate_transactions() -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(RNG_SEED)
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


def main() -> None:
    print("=== Fraud Spike MVP: generating synthetic merchant transactions ===")
    transactions, injections = simulate_transactions()
    print(f"Generated {len(transactions):,} transactions across {transactions.merchant_id.nunique()} merchants over {HOURS} hours.")
    print("Injected windows by merchant/scenario:")
    print(pd.DataFrame(injections).groupby(["merchant_id", "label"]).size().unstack(fill_value=0).to_string())
    print("Scenario row counts:", transactions["scenario_label"].value_counts().to_dict())

    features = build_features(transactions)
    split_time = features["window_start"].min() + pd.Timedelta(hours=HOURS * 0.70)
    train, test = features[features["window_start"] < split_time].copy(), features[features["window_start"] >= split_time].copy()
    print("\n=== Rolling 30-second feature windows ===")
    print(f"Created {len(features)} windows: {len(train)} train / {len(test)} test.")
    print("Candidate spikes by split:", {"train": int(train.candidate_spike.sum()), "test": int(test.candidate_spike.sum())})

    train_candidates = train[train["candidate_spike"] & train["majority_scenario_label"].isin(["card_testing", "demand_spike"])].copy()
    print("\n=== Stage 2 training ===")
    print(f"Eligible train candidate windows: {len(train_candidates)}; labels: {train_candidates.majority_scenario_label.value_counts().to_dict()}")
    if train_candidates["majority_scenario_label"].nunique() < 2:
        raise RuntimeError("Training data lacks both injected spike classes; rerun with a different seed.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=RNG_SEED),
    )
    model.fit(train_candidates[FEATURE_COLUMNS], train_candidates["majority_scenario_label"])
    print("Trained LogisticRegression on:", ", ".join(FEATURE_COLUMNS))

    test["predicted_label"] = "normal"
    test["card_testing_probability"] = 0.0
    test_candidates = test[test["candidate_spike"]].copy()
    if not test_candidates.empty:
        probabilities = model.predict_proba(test_candidates[FEATURE_COLUMNS])
        card_index = list(model.classes_).index("card_testing")
        test.loc[test_candidates.index, "card_testing_probability"] = probabilities[:, card_index]
        test.loc[test_candidates.index, "predicted_label"] = model.predict(test_candidates[FEATURE_COLUMNS])
        print("Test candidate card-testing probability range: "
              f"min={probabilities[:, card_index].min():.3f}, max={probabilities[:, card_index].max():.3f}")
    alerts = test[(test["predicted_label"] == "card_testing") & (test["card_testing_probability"] > 0.60)].copy()

    print("\n=== Alerts fired (test candidate windows only) ===")
    if alerts.empty:
        print("No card-testing alerts met the > 0.60 confidence threshold.")
    for _, alert in alerts.iterrows():
        detail = ", ".join(f"{feature}={alert[feature]:.3f}" for feature in FEATURE_COLUMNS)
        print(f"ALERT merchant={alert.merchant_id} window={alert.window_start} to {alert.window_start + pd.Timedelta(seconds=WINDOW_SECONDS)} "
              f"type=card_testing confidence={alert.card_testing_probability:.3f} | {detail}")

    print("\n=== Full test-set evaluation ===")
    truth = test["majority_scenario_label"].eq("card_testing")
    predicted = test.index.isin(alerts.index)
    precision = precision_score(truth, predicted, zero_division=0)
    recall = recall_score(truth, predicted, zero_division=0)
    f1 = f1_score(truth, predicted, zero_division=0)
    print(f"Card-testing precision: {precision:.3f}")
    print(f"Card-testing recall:    {recall:.3f}")
    print(f"Card-testing F1:        {f1:.3f}")
    demand = test["majority_scenario_label"].eq("demand_spike")
    fp_rate = float(predicted[demand].mean()) if demand.any() else float("nan")
    print(f"False-positive rate on demand_spike windows: {fp_rate:.3f}")
    non_attack = ~truth
    fp_all_rate = float(predicted[non_attack].mean()) if non_attack.any() else float("nan")
    print(f"False-positive rate on all non-attack windows: {fp_all_rate:.3f}")
    print(f"Test-set cardinality: {int(truth.sum())} card-testing windows; {int(demand.sum())} demand-spike windows; "
          f"{int((~truth & ~demand).sum())} quiet-period windows")
    print("Detection latency for injected card-testing attacks:")
    attack_level_results = []
    for attack in (item for item in injections if item["label"] == "card_testing"):
        matching = alerts[(alerts["merchant_id"] == attack["merchant_id"])
                          & (alerts["window_start"] >= attack["start"])
                          & (alerts["window_start"] < attack["end"])]
        if matching.empty:
            scope = "outside test split" if attack["end"] <= test["window_start"].min() else "no alert"
            print(f"  {attack['merchant_id']} {attack['start']}: {scope}")
            attack_level_results.append({"merchant_id": attack["merchant_id"], "attack_start": attack["start"], "detected": False, "latency_min": None})
        else:
            latency = (matching["window_start"].iloc[0] - attack["start"]).total_seconds() / 60
            print(f"  {attack['merchant_id']} {attack['start']}: {latency:.2f} minutes")
            attack_level_results.append({"merchant_id": attack["merchant_id"], "attack_start": attack["start"], "detected": True, "latency_min": latency})
    detected_latencies = [item["latency_min"] for item in attack_level_results if item["detected"]]
    print(f"Attack-level detection (test alerts): {len(detected_latencies)}/{len(attack_level_results)} attacks detected")
    test_attacks = [item for item in attack_level_results if item["attack_start"] >= test["window_start"].min()]
    test_detected = sum(item["detected"] for item in test_attacks)
    print(f"Holdout attack coverage: {test_detected}/{len(test_attacks)} test-set attacks detected (small sample)")
    median_latency = float(median(detected_latencies)) if detected_latencies else None
    print(f"Median detection latency: {median_latency:.2f} minutes" if median_latency is not None
          else "Median detection latency: no detected attacks")

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "attacks_detected": f"{test_detected}/{len(test_attacks)}",
        "median_latency_min": median_latency,
        "fp_rate_demand_spike": fp_rate,
        "fp_rate_all_nonattack": fp_all_rate,
        "test_window_counts": {
            "card_testing": int(truth.sum()),
            "demand_spike": int(demand.sum()),
            "quiet_period": int((~truth & ~demand).sum()),
        },
        "alerts": [
            {
                "merchant_id": alert.merchant_id,
                "window_start": str(alert.window_start),
                "window_end": str(alert.window_start + pd.Timedelta(seconds=WINDOW_SECONDS)),
                "confidence": float(alert.card_testing_probability),
                "spike_type": "card_testing",
            }
            for _, alert in alerts.iterrows()
        ],
        "velocity_series": [
            {"merchant_id": row.merchant_id, "window_start": str(row.window_start), "velocity": float(row.velocity)}
            for _, row in features.iterrows()
        ],
        "injections": [
            {"merchant_id": item["merchant_id"], "label": item["label"], "start": str(item["start"]), "end": str(item["end"])}
            for item in injections
        ],
    }
    with open("results.json", "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
    print("Results exported to results.json")

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
    plt.title(f"{display_merchant} velocity: injected spikes and fired alerts")
    plt.xlabel("Window start")
    plt.ylabel("Transactions per second")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spike_detection_demo.png", dpi=150)
    print("Saved spike_detection_demo.png")


if __name__ == "__main__":
    main()
