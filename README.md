# Fraud Triage Service — Razorpay Merchant Risk Desk (synthetic capstone)

## Overview

This service decides, before settlement, whether a payment is approved or held for a human analyst.
Part A cleans the raw payment export and adds three engineered features.
Part B trains a class-balanced random forest and picks an alert threshold that keeps validation recall at or above 0.65.
Part C retrieves the fraud policy that fits a flagged payment using TF-IDF retrieval.
It then asks a language model (the offline `mock_llm.py`) for a JSON case note and checks that note in three gates.
Failed notes are retried, then replaced by a safe default, and a guardrail stops any "approve" on a risky payment.
Part D exposes everything as a FastAPI service with `/score`, `/case`, `/health` and `/metrics`.
The service has input guardrails (400 and 422), an output guardrail for large new-device payments, and 503 on a model outage.
It writes one log line per POST request and keeps live counters.
All data is synthetic and fictional; amounts are in INR.

## Reproduce

Run every command from the repository root (Python 3.11 or newer).

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py              # prints: rows written: 6090 (files are also committed)
python prepare.py                    # Part A audit and checkpoints
python train.py                      # Part B tables (about 30 seconds)
python retrieval.py                  # Part C retrieval table
python casenote.py                   # Part C scenario table
rm -f service.log                    # Windows: del service.log
uvicorn app:app --port 8000          # terminal 1: start a fresh service (trains at start-up, about 15 s)
python smoke_test.py                 # terminal 2: eleven requests, result table and /metrics
```

Stop the service with Ctrl+C after the smoke test.
Restart it before any further smoke run, because the mock model remembers call counts per transaction id.

## Part A results

| Audit item | Value |
|---|---|
| Rows / columns as read | 6,090 / 14 |
| Exact duplicate rows | 90 |
| Rows after de-duplication | 6,000 |
| Missing `amount_inr` (after de-dup) | 76 |
| Missing `distance_from_home_km` (after de-dup) | 193 |
| `amount_inr` below 1 or above 1,000,000 (after de-dup) | 5 |
| Distinct raw spellings: `merchant_category` / `city` | 24 / 24 |

**Cleaned table.**

| Item | Value |
|---|---|
| Rows | 5,919, with no duplicates |
| `merchant_category` | Grocery 1,406; Food Delivery 1,326; Fashion 976; Electronics 845; Utilities 685; Travel 681 |
| `city` | 6 values |
| `payment_mode` | UPI 3,221; Card 1,450; NetBanking 731; Wallet 517 |
| `amount_inr` | min 141.37, max 105,883.95 |
| Missing `distance_from_home_km` remaining | 191 |

**Engineered features.**

| Feature | Value |
|---|---|
| `is_night` sum | 365 |
| `high_value` sum | 17 |
| `amount_vs_avg_ratio` max | 13.11 |

Both functions run on a one-row DataFrame, and `clean_transactions` leaves its input unchanged (both checked by `python prepare.py`).

**Why distance is not filled here.** The missing `distance_from_home_km` values are left empty in Part A because the fill value (a median) must be learned from the training rows only, otherwise validation and test information leaks into training.

**Leakage audit (cleaned table).**

| chargeback_filed | Rows | Fraud rows |
|---|---|---|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |

`chargeback_filed` is recorded only after a payment has settled and the customer disputes it, so it is not known when the approve/hold decision is made.
Feeding it to the model would let it "predict" fraud from the outcome itself: 304 of 373 chargeback rows are fraud.
`txn_id` and `customer_id` are arbitrary identifiers with no causal link to fraud, so a model could only memorise them, and they would not generalise to new payments.

Excluded columns: `txn_id`, `customer_id`, `chargeback_filed`, `is_fraud` (the target).

Model-input columns (before one-hot): `merchant_category`, `city`, `payment_mode`, `amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`, `amount_vs_avg_ratio`, `is_night`, `high_value`.

**Why accuracy misleads.** There are 366 fraud rows out of 5,919 (6.18%). A "never fraud" rule scores accuracy 93.82% with recall 0.
Because 93.82% of payments are legitimate, a model can look highly accurate while catching no fraud at all.
The risk desk should therefore lead with recall, precision and F1 on the fraud class.

## Part B results

**Input columns (26, pinned order).**

```
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card', 'payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']
```

**Split** (stratified 60/20/20, `random_state=42`).

| Split | Rows | Fraud |
|---|---|---|
| Train | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training median of `distance_from_home_km` is 12.6, used to fill all three splits.

**Imbalance treatments** (scaled logistic regression, validation split, threshold 0.5).

| Variant | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Plain | 0.625 | 0.342 | 0.442 | 1096 | 15 | 48 | 25 |
| class_weight="balanced" | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| Random over-sampling | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The over-sampled training table has 6,662 rows (3,331 non-fraud + 3,331 resampled fraud).

A risk desk that must not miss fraud would prefer a balanced or over-sampled model: recall rises from 0.342 to 0.781, so missed frauds fall from 48 to 16.
The cost is precision dropping from 0.625 to about 0.25, so held legitimate payments grow from 15 to about 165–169, each needing an analyst's time.

**Cross-validation on the training split** (StratifiedKFold, 5 folds, F1).

| Model | Mean F1 | Std |
|---|---|---|
| Balanced logistic regression | 0.402 | 0.015 |
| Random forest, untuned (balanced) | 0.453 | 0.052 |
| Random forest, tuned (best of 12 × 5 folds) | 0.465 | 0.042 |

Best parameters: `max_depth=None`, `min_samples_leaf=5` (with `n_estimators=200`, `class_weight="balanced"` and `random_state=42` fixed).

**Choice of the final model.** The tuned forest has the best accuracy on the training folds, with mean CV F1 0.465 against 0.402 for balanced logistic regression.
Its stability is acceptable: the fold spread of 0.042 is wider than logistic regression's 0.015, but the gap in means is larger than that spread.
Interpretability comes from feature importances, which point to plausible fraud signals (spend ratio, velocity), although single decisions are harder to explain.
What we give up relative to logistic regression is a simple per-feature coefficient story, tighter fold-to-fold stability, and a model that is cheap to audit.

**Threshold table** (tuned forest, validation split).

| Threshold | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.1 | 0.111 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

**t_star = 0.4**: the highest threshold whose validation recall (0.712) is at least 0.65. At 0.5, recall is only 0.589.

**Final test at t_star = 0.4** (test split used once).

| Precision | Recall | F1 | Flagged | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| 0.281 | 0.781 | 0.413 | 203 | 965 | 146 | 16 | 57 |

**Risk-committee note.** The alert threshold is 0.4, chosen as the highest validation threshold that still catches at least 65% of fraud.
On the 1,184 test payments the model flagged 203 for review, catching 57 of 73 frauds and missing 16.
It held 146 legitimate payments, about 2.56 legitimate holds per fraud caught.
A stricter threshold of 0.5 would cut validation flags from 191 to 122 and raise precision from 0.272 to 0.352.
However, it would drop recall from 0.712 to 0.589, missing roughly one more fraud in every eight.

**Leakage experiment** (validation F1 at 0.5).

| Model | F1 |
|---|---|
| Clean tuned forest (26 columns) | 0.441 |
| Leaky forest (27 columns, `chargeback_filed` last) | 0.797 |

The jump of 0.356 comes from a column that is only known after settlement, so the model is reading the answer rather than predicting it.
In live triage that value does not exist yet, so the 0.797 would collapse to the clean score and give the risk desk false confidence.

**Top five feature importances** (tuned forest).

| Feature | Importance |
|---|---|
| amount_vs_avg_ratio | 0.270 |
| txn_count_last_24h | 0.178 |
| amount_inr | 0.134 |
| account_age_days | 0.084 |
| distance_from_home_km | 0.071 |

A fraudster with a stolen instrument spends far above the real owner's usual amount (high `amount_vs_avg_ratio`).
They also try to cash out quickly with many payments in a short window (high `txn_count_last_24h`).

## Part C results

**Retrieval** (TF-IDF with default settings over the six policy texts, cosine similarity).

| Query | Expected | Retrieved | Score |
|---|---|---|---|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | none | P3 | 0.4251 |
| how do I reset my email password | none | P1 | 0.0000 |

In-scope top-1 hits: **8 of 8**.

`MIN_SCORE = 0.45`. It sits above the highest out-of-scope score (0.4251) and at or below the lowest in-scope score (0.4862), so both out-of-scope queries return `no_policy_match` and never reach the model.

**Scenario run** (`python casenote.py`, after `mock_llm.reset()`).

| Scenario | Retrieved policy | Attempts | Failures | Status | Safe default | Final action | Override |
|---|---|---|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals:
- 19 model calls in all.
- 3 scenarios valid on the first call (S1, S6, S8).
- 6 recovered by retry (S2, S3, S4, S7, S9, S10).
- 1 safe default (S5).
- 1 guardrail override (S6).

Further checks (run by `python casenote.py`):
- `run_case` for T900099 raises `ConnectionError`.
- A no-match question returns `no_policy_match`, attempts 0 and `case_note` None.
- `validate` returns gates 1, 2, 3 and 0 for non-JSON, a missing key, a wrong policy id and a good note.

A hosted model samples its output, can be served by different replicas or versions, and is affected by load and truncation, so the same prompt can fail twice in a row.
Downstream systems cannot trust any single reply, so every note must be parsed and validated against the schema, retried a bounded number of times, and replaced by a safe default when it never passes.

## Part D results

**Smoke test** (fresh `uvicorn app:app --port 8000`, then `python smoke_test.py`).

| Id | Category | Expected | Observed | Result |
|---|---|---|---|---|
| Q1 | happy | 200 | 200 | PASS |
| Q2 | happy | 200 | 200 | PASS (p=0.043) |
| Q3 | escalation | 200 | 200 | PASS (p=0.9597) |
| Q4 | escalation | 200 | 200 | PASS (p=0.1765) |
| Q5 | edge | 400 | 400 | PASS |
| Q6 | edge | 422 | 422 | PASS |
| Q7 | escalation | 200 | 200 | PASS (p=0.9597) |
| Q8 | escalation | 200 | 200 | PASS (p=0.9597) |
| Q9 | edge | 503 | 503 | PASS |
| Q10 | happy | 200 | 200 | PASS (p=0.043) |
| Q11 | edge | 200 | 200 | PASS (p=0.9597) |

By category: happy 3/3, escalation 4/4, edge 4/4, total 11/11.

**/metrics after that run.**

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 18.07,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

**Three lines from service.log** (the committed file holds exactly the ten POST lines of that run).

```
2026-10-19 14:52:29,936 INFO POST /score status=200 latency_ms=27.3 txn_id=T800001 fraud_probability=0.043 decision=approve human_review=False
2026-10-19 14:52:29,984 WARNING POST /score status=200 latency_ms=25.8 txn_id=T800003 fraud_probability=0.1765 decision=hold_for_review human_review=True guardrail_override=True
2026-10-19 14:52:30,058 ERROR POST /case status=503 latency_ms=19.0 txn_id=T900099 fraud_probability=0.9597 reason=llm_unavailable
```

**Extra check** (separate fresh service, not included in the log or metrics above). `POST /case` for T900006 with the brief's body gave:
- Status 200 and probability 0.4391, so the payment is flagged but below 0.5.
- `decision` hold_for_review, with `case_note.recommended_action` hold_for_review.
- `guardrail_override` true and `human_review` true.

## Limitations

- The data is synthetic, generated from a hand-built fraud formula, so the model's performance says nothing about real payment fraud.
- The threshold was tuned on only 73 validation fraud cases, so `t_star` and the precision/recall figures carry wide uncertainty.
- The TF-IDF retriever matches words, not meaning, so a paraphrased question with different vocabulary can miss its policy or match the wrong one (the cafeteria query scored 0.4251, close to the cut-off).
- The language model is a scripted offline stand-in, so retry rates and failure modes of a real hosted model are not measured.
- Metrics are in-memory and per process, so they reset on restart and would not aggregate across several service workers.
- The model is retrained at every start-up rather than versioned and loaded, so there is no model registry or rollback.
