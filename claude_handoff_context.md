# Context handoff: Fraud Spike Detector (Razorpay AI Buildathon, Track 2)

Paste this whole thing as your first message to bring me up to speed instantly.

---

## What this is
I'm building a submission for the **Razorpay AI Buildathon** — a student-only hackathon
where winners get a paid 6-12 month AI Builder Internship in Bangalore. I picked
**Track 2: AI Risk Manager** ("Stop the merchant losing money to fraud, returns and
chargebacks. Build a working detector, verifier or auto-responder for one class of loss,
with measured precision and recall on a held-out test set."). The bar for this track is
explicitly: **honest metrics including false-positive cost, strictly defense-only**
(anything offense-capable is disqualified).

Deadline: submitting **tonight**. Priority is a working, honestly-evaluated prototype
over polish or completeness.

## The problem I chose: fraud-spike detector, framed around a real gap
Not a generic "classify this transaction as fraud" model (that's the commodity Kaggle
approach). The actual angle: **chargeback labels take 30-120 days to arrive, so a
classifier trained on confirmed fraud labels is structurally blind to a NEW attack
happening right now.** The real, underserved problem is detecting that a merchant is
CURRENTLY under a coordinated **card-testing attack** — in real time, from behavioral
pattern alone, before any confirmed label exists — and telling it apart from a
legitimate demand spike (e.g. a flash sale), since both look like "sudden volume
increase" on the surface.

**Card-testing attack pattern**: fraudsters run stolen card numbers through checkout in
small low-value transactions to find which ones still work. Signature: burst of
transactions, high decline rate, narrow BIN pool (one leaked batch), narrow device/IP
pool (a few proxies), low transaction amounts.

**Demand-spike pattern (the "don't cry wolf" control)**: also a sudden volume burst, but
high approval rate, wide card/device/IP diversity, normal-to-higher amounts (real
customers, real purchases).

## Architecture (built and working)
```
Transaction simulator (synthetic, labeled)
  -> Rolling feature engine (per-merchant sliding windows: velocity, decline_rate,
     bin_diversity_ratio, device_diversity_ratio, ip_diversity_ratio, pct_low_amount,
     mean_amount)
  -> Stage 1: change-point detector (rolling mean/std z-score threshold on velocity,
     BASELINE FREEZES while a spike is active so a sustained attack doesn't get
     absorbed into "normal" and stop triggering — this was a real bug we found and fixed)
  -> Stage 2: lightweight classifier (sklearn LogisticRegression) on flagged windows
     only, using decline_rate/diversity ratios/amount features, predicting
     card_testing vs demand_spike
  -> Alert engine: emits alert with merchant, window, confidence, contributing features,
     suggested bounded action (e.g. "hold approvals for BIN range X")
  -> Evaluation harness: reports BOTH window-level precision/recall/F1 AND attack-level
     detection (did we catch each true injected attack at all, and how fast — this is
     the metric that actually matters operationally, not window-level recall, since a
     5-minute attack spans many windows and catching the attack once is what counts)
```

## Current file
`fraud_spike_mvp.py` — single-file MVP (deliberately not split into a package structure;
this is a hackathon MVP, not production code). Also generates `spike_detection_demo.png`
(velocity-over-time plot with true attack starts marked and fired alerts marked).

Data is entirely synthetic (no public dataset fits — real card-testing attack windows
are proprietary to payment processors and don't exist in any public dataset). Baseline
transaction statistics (amount distribution, decline rate) should be loosely calibrated
against public dataset stats (e.g. Kaggle Credit Card Fraud dataset) rather than
invented from nothing, to make the simulation defensible rather than arbitrary.

## Latest validated run
```
Candidate spikes: {'train': 30, 'test': 46}
Card-testing precision: 1.000
Card-testing recall:    1.000
Card-testing F1:        1.000
False-positive rate on demand_spike windows: 0.000
Attack-level detection (test alerts): 2/3 attacks detected
Median detection latency: 0.00 minutes
```
NOTE: only 3 total injected card-testing attacks exist across train+test (1 landed in
train, 2 in test) and 3 demand spikes. These are TINY sample sizes — perfect metrics on
n=2 test attacks are not statistically meaningful and must be reported with that caveat,
not presented as if they prove the model works reliably. This is explicitly a track
requirement ("honest metrics") — a suspiciously perfect score with no caveat reads worse
to judges than an honest limitation paragraph.

## What's left before submission tonight
1. Sanity-check the "perfect" metrics aren't a leakage/degenerate-threshold artifact
   given the tiny sample (see caveat above) — possibly increase injected attack/spike
   count (e.g. 8-10 each instead of 3) if there's time, purely to make the evaluation
   less trivially small, without touching the core pipeline logic.
2. Check false-positive rate against NORMAL (non-injected) windows too, not just
   demand_spike windows — currently only demand_spike FPR is reported; a spurious
   Stage-1 trigger on an ordinary quiet period that Stage 2 also misclassifies as
   card_testing wouldn't show up in the current FPR metric.
3. Write README.md: problem framing (label-latency gap), architecture diagram (ASCII is
   fine), how to run, the real evaluation numbers pasted in verbatim, an explicit "known
   limitations" section (small sample size, single merchant, synthetic data, simplified
   Stage 1 vs a true online EWMA/CUSUM), and an explicit defense-only statement (required
   by the track rules).
4. Generate a short concept demo video (already have a Veo/Gemini prompt drafted for
   this) to accompany the pitch, since Razorpay wants a public repo + 5-minute pitch
   video + architecture explanation.
5. Submit.

## What I need help with right now
[Fill this in with your actual next ask before pasting — e.g. "help me write the README"
or "help me add the normal-window false-positive check" or "review my pitch video script"]
