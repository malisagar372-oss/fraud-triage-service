"""app.py - Part D: the Fraud Triage Service.

    uvicorn app:app --port 8000
"""
import logging
import threading
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casenote import run_case
from prepare import add_features, normalise_text
from train import build_final_model

# ---------------------------------------------------------------- start-up
MODEL = build_final_model()              # trained once, kept in memory, never saved to disk
FOREST, COLUMNS = MODEL["model"], MODEL["columns"]
MEDIAN, THRESHOLD = MODEL["median"], MODEL["threshold"]

KNOWN_CATEGORIES = {c.removeprefix("merchant_category_") for c in COLUMNS
                    if c.startswith("merchant_category_")}
LARGE_PAYMENT_INR = 100_000

log = logging.getLogger("triage")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.FileHandler("service.log")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)

app = FastAPI(title="Fraud Triage Service")


# ---------------------------------------------------------------- request models
class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


# ---------------------------------------------------------------- metrics
_lock = threading.Lock()
_counters = {"requests_total": 0, "success_total": 0, "error_total": 0,
             "latency_ms_sum": 0.0, "human_review_total": 0}


def _record(status, ms, human_review):
    with _lock:
        _counters["requests_total"] += 1
        _counters["success_total" if status < 400 else "error_total"] += 1
        _counters["latency_ms_sum"] += ms
        if status < 400 and human_review:
            _counters["human_review_total"] += 1


@app.middleware("http")
async def observe(request: Request, call_next):
    """The one place that logs and counts every POST response (200, 400, 422, 503 alike)."""
    t0 = time.perf_counter()
    request.state.info = {}
    response = await call_next(request)
    if request.method == "POST":
        ms = (time.perf_counter() - t0) * 1000
        info = request.state.info
        status = response.status_code
        _record(status, ms, info.get("human_review", False))

        parts = [f"{request.method} {request.url.path}", f"status={status}", f"latency_ms={ms:.1f}"]
        for key in ("txn_id", "fraud_probability", "decision", "human_review",
                    "safe_default", "guardrail_override", "reason"):
            if key in info:
                parts.append(f"{key}={info[key]}")
        if status >= 500:
            level = logging.ERROR
        elif status >= 400 or info.get("safe_default") or info.get("guardrail_override"):
            level = logging.WARNING
        else:
            level = logging.INFO
        log.log(level, " ".join(parts))
    return response


# ---------------------------------------------------------------- scoring path
def check_input(t: Txn):
    """Return a 400 JSONResponse for an impossible value, else None. Runs before any model."""
    category = t.merchant_category.strip().title()
    problems = [
        (not 1 <= t.amount_inr <= 1_000_000, "amount_inr", "amount_inr must be between 1 and 1,000,000"),
        (not 0 <= t.hour_of_day <= 23, "hour_of_day", "hour_of_day must be between 0 and 23"),
        (t.txn_count_last_24h < 0, "txn_count_last_24h", "txn_count_last_24h must not be negative"),
        (t.avg_amount_30d_inr <= 0, "avg_amount_30d_inr", "avg_amount_30d_inr must be greater than 0"),
        (category not in KNOWN_CATEGORIES, "merchant_category",
         f"merchant_category must be one of {sorted(KNOWN_CATEGORIES)}"),
    ]
    for bad, field, message in problems:
        if bad:
            return JSONResponse(status_code=400, content={"error": message, "field": field})
    return None


def fraud_probability(t: Txn):
    row = pd.DataFrame([t.model_dump(exclude={"question"})]).drop(columns=["txn_id"])
    row["distance_from_home_km"] = pd.to_numeric(row["distance_from_home_km"]).astype(float)
    row = add_features(normalise_text(row))
    X = pd.get_dummies(row, dtype=int).reindex(columns=COLUMNS, fill_value=0)
    X["distance_from_home_km"] = X["distance_from_home_km"].fillna(MEDIAN)
    return float(FOREST.predict_proba(X)[0, 1])


def score_txn(t: Txn, info):
    """Shared by /score and /case: probability, flag, decision (with the large-payment hold)."""
    prob = round(fraud_probability(t), 4)
    flagged = prob >= THRESHOLD
    decision = "hold_for_review" if flagged else "approve"
    large_hold = (not flagged and t.amount_inr >= LARGE_PAYMENT_INR and t.is_new_device == 1)
    if large_hold:
        decision = "hold_for_review"
    info.update(fraud_probability=prob)
    return prob, flagged, decision, large_hold


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with _lock:
        c = dict(_counters)
    n = c["requests_total"]
    rate = lambda x: round(x / n, 4) if n else 0.0
    return {"requests_total": n, "success_total": c["success_total"], "error_total": c["error_total"],
            "success_rate": rate(c["success_total"]), "error_rate": rate(c["error_total"]),
            "avg_latency_ms": round(c["latency_ms_sum"] / n, 2) if n else 0.0,
            "human_review_total": c["human_review_total"],
            "human_review_rate": rate(c["human_review_total"])}


@app.post("/score")
def score(t: Txn, request: Request):
    info = request.state.info
    info["txn_id"] = t.txn_id
    if (bad := check_input(t)) is not None:
        info["reason"] = "input_rejected"
        return bad
    prob, flagged, decision, large_hold = score_txn(t, info)
    human_review = decision == "hold_for_review"
    info.update(decision=decision, human_review=human_review)
    if large_hold:
        info["guardrail_override"] = True
    return {"txn_id": t.txn_id, "fraud_probability": prob, "flagged": flagged,
            "decision": decision, "human_review": human_review}


@app.post("/case")
def case(c: CaseIn, request: Request):
    info = request.state.info
    info["txn_id"] = c.txn_id
    if (bad := check_input(c)) is not None:
        info["reason"] = "input_rejected"
        return bad
    prob, flagged, decision, large_hold = score_txn(c, info)
    case_note, attempts, safe, override = None, 0, False, large_hold

    if flagged:
        try:
            r = run_case(c.txn_id, prob, c.amount_inr, c.question)
        except ConnectionError:
            info["reason"] = "llm_unavailable"
            return JSONResponse(status_code=503, content={
                "error": "case-note service temporarily unavailable, try again later"})
        attempts, safe = r["attempts"], r["safe_default"]
        override = override or r["guardrail_override"]
        if r["status"] == "no_policy_match":
            decision = "hold_for_review"
        else:
            case_note = r["case_note"]
            if case_note["recommended_action"] == "approve":      # flagged payment: never auto-approve
                case_note["recommended_action"] = "hold_for_review"
                override = True
            decision = case_note["recommended_action"]

    human_review = decision in ("hold_for_review", "block_and_notify") or safe
    info.update(decision=decision, human_review=human_review)
    if safe:
        info["safe_default"] = True
    if override:
        info["guardrail_override"] = True
    return {"txn_id": c.txn_id, "fraud_probability": prob, "flagged": flagged,
            "decision": decision, "human_review": human_review, "case_note": case_note,
            "attempts": attempts, "safe_default": safe, "guardrail_override": override}
