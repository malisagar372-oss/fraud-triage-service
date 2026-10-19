"""smoke_test.py - Part D: sends the eleven requests Q1-Q11 to a running service.

Start a fresh service first:  uvicorn app:app --port 8000
Then:                         python smoke_test.py
"""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"

BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
          "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
          "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900}
RISKY = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
         "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
         "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40}
BIG = {**BENIGN, "merchant_category": "Utilities", "payment_mode": "NetBanking", "amount_inr": 150000,
       "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0, "is_new_device": 1,
       "account_age_days": 2200}


def body(template, txn_id, question=None, drop=None, **changes):
    b = {**template, **changes, "txn_id": txn_id}
    if question is not None:
        b["question"] = question
    if drop:
        b.pop(drop)
    return b


# (id, category, method, path, json body, expected status, expected fields)
REQUESTS = [
    ("Q1", "happy", "GET", "/health", None, 200, {"status": "ok"}),
    ("Q2", "happy", "POST", "/score", body(BENIGN, "T800001"), 200,
     {"flagged": False, "decision": "approve", "human_review": False}),
    ("Q3", "escalation", "POST", "/score", body(RISKY, "T800002"), 200,
     {"flagged": True, "decision": "hold_for_review", "human_review": True}),
    ("Q4", "escalation", "POST", "/score", body(BIG, "T800003"), 200,
     {"flagged": False, "decision": "hold_for_review", "human_review": True}),
    ("Q5", "edge", "POST", "/score", body(BENIGN, "T800004", amount_inr=-5), 400,
     {"field": "amount_inr", "error": ...}),
    ("Q6", "edge", "POST", "/score", body(BENIGN, "T800005", drop="hour_of_day"), 422, {}),
    ("Q7", "escalation", "POST", "/case",
     body(RISKY, "T900002", "more than 6 UPI payments in 24 hours velocity alert"), 200,
     {"flagged": True, "attempts": 2, "safe_default": False, "case_note.policy_id": "P2",
      "human_review": True}),
    ("Q8", "escalation", "POST", "/case",
     body(RISKY, "T900005", "electronics merchant payment on an account younger than 30 days"), 200,
     {"attempts": 3, "safe_default": True, "decision": "hold_for_review", "human_review": True}),
    ("Q9", "edge", "POST", "/case",
     body(RISKY, "T900099", "card payment above INR 50000 from a new device"), 503, {"error": ...}),
    ("Q10", "happy", "POST", "/case",
     body(BENIGN, "T800001", "card payment above INR 50000 from a new device"), 200,
     {"case_note": None, "attempts": 0, "decision": "approve", "human_review": False}),
    ("Q11", "edge", "POST", "/case", body(RISKY, "T800006", "what is the cafeteria menu today"), 200,
     {"case_note": None, "attempts": 0, "decision": "hold_for_review", "human_review": True}),
]


def lookup(data, dotted):
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return "<missing>"
        data = data[part]
    return data


def wait_for_service(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(BASE + "/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def main():
    if not wait_for_service():
        print("service did not come up within 120 seconds")
        sys.exit(1)

    results = []
    print(f"{'id':<5}{'category':<12}{'expected':>9}{'observed':>9}  result  notes")
    for rid, cat, method, path, payload, exp_status, exp_fields in REQUESTS:
        r = requests.request(method, BASE + path, json=payload, timeout=60)
        try:
            data = r.json()
        except ValueError:
            data = {}
        problems = [] if r.status_code == exp_status else [f"status {r.status_code}"]
        for key, want in exp_fields.items():
            got = lookup(data, key)
            if want is ...:                       # field must be present
                if got == "<missing>":
                    problems.append(f"{key} missing")
            elif got != want:
                problems.append(f"{key}={got!r} (want {want!r})")
        ok = not problems
        results.append((cat, ok))
        note = "; ".join(problems) if problems else (
            f"p={data['fraud_probability']}" if "fraud_probability" in data else "")
        print(f"{rid:<5}{cat:<12}{exp_status:>9}{r.status_code:>9}  {'PASS' if ok else 'FAIL':<6}  {note}")

    print()
    for cat in ("happy", "escalation", "edge"):
        got = [ok for c, ok in results if c == cat]
        print(f"{cat:<11} {sum(got)}/{len(got)}")
    total = sum(ok for _, ok in results)
    print(f"{'total':<11} {total}/{len(results)}")

    print("\n/metrics:")
    print(json.dumps(requests.get(BASE + "/metrics", timeout=5).json(), indent=2))
    sys.exit(0 if total == len(results) else 1)


if __name__ == "__main__":
    main()
