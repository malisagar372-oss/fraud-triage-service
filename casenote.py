"""casenote.py - Part C: policy-grounded case notes, three validation gates, retry and guardrail.

    from casenote import build_prompt, validate, run_case
    python casenote.py     # runs the 10 scenarios and prints the table
"""
import json

import mock_llm
from retrieval import MIN_SCORE, retrieve

# The case-note rules, written once in JSON Schema vocabulary; every gate reads from here.
CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
}
TYPE_MAP = {"object": dict, "string": str}

SAFE_SUMMARY = "Automated case note unavailable; manual review required."


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    props = CASE_NOTE_SCHEMA["properties"]
    allowed = "\n".join(
        f'- "{k}": ' + (" | ".join(props[k]["enum"]) if "enum" in props[k] else "one or two plain sentences")
        for k in CASE_NOTE_SCHEMA["required"])
    return (
        "ROLE\n"
        "You are a fraud analyst assistant on the Merchant Risk Desk.\n\n"
        "TASK\n"
        "Write a short case note for the held payment below, citing the fraud policy that applies.\n\n"
        "CONSTRAINTS\n"
        "- Use only the policy text given below; do not add rules or facts that are not in it.\n"
        "- Reply with one JSON object and nothing else: no prose, no markdown, no code fences.\n"
        f"- policy_id must be {policy_id}.\n\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n"
        f"POLICY_TEXT: {policy_text}\n\n"
        "OUTPUT FORMAT\n"
        "A JSON object with exactly these five string keys and allowed values:\n"
        f"{allowed}\n"
    )


def validate(text, retrieved_policy_id):
    """Return (note, gate). gate 0 = passed, else the first gate that failed."""
    # Gate 1: valid JSON and a JSON object
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1
    if not isinstance(note, TYPE_MAP[CASE_NOTE_SCHEMA["type"]]):
        return None, 1
    # Gate 2: every required key present with the declared type
    props = CASE_NOTE_SCHEMA["properties"]
    for key in CASE_NOTE_SCHEMA["required"]:
        if key not in note or not isinstance(note[key], TYPE_MAP[props[key]["type"]]):
            return note, 2
    # Gate 3: values inside the allowed lists, and the policy id matches retrieval
    for key, rule in props.items():
        if "enum" in rule and note[key] not in rule["enum"]:
            return note, 3
    if note["policy_id"] != retrieved_policy_id:
        return note, 3
    return note, 0


def safe_default(policy_id):
    return {"summary": SAFE_SUMMARY, "priority": "high", "recommended_team": "customer_care",
            "recommended_action": "hold_for_review", "policy_id": policy_id}


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    result = {"status": "ok", "attempts": 0, "failures": [],
              "case_note": None, "safe_default": False, "guardrail_override": False}
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        result["status"] = "no_policy_match"
        return result
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)

    note = None
    for _ in range(max_attempts):
        text = mock_llm.generate(prompt)          # ConnectionError propagates on purpose
        result["attempts"] += 1
        parsed, gate = validate(text, policy_id)
        if gate == 0:
            note = parsed
            break
        result["failures"].append(gate)
    if note is None:
        note = safe_default(policy_id)
        result["safe_default"] = True

    # Output guardrail: never approve a payment with fraud probability >= 0.5
    if note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        note["recommended_action"] = "hold_for_review"
        result["guardrail_override"] = True
    result["case_note"] = note
    return result


def main():
    mock_llm.reset()
    with open("case_scenarios.json") as f:
        scenarios = json.load(f)
    yn = lambda b: "yes" if b else "no"
    print(f"{'scenario':<14}{'policy':<8}{'attempts':<10}{'failures':<12}{'status':<8}"
          f"{'safe default':<14}{'final action':<18}{'override'}")
    calls = first = retried = safe = overrides = 0
    for i, s in enumerate(scenarios, 1):
        r = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        calls += r["attempts"]
        first += r["attempts"] == 1 and not r["failures"]
        retried += bool(r["failures"]) and not r["safe_default"]
        safe += r["safe_default"]
        overrides += r["guardrail_override"]
        note = r["case_note"] or {}
        label = f"S{i} ({s['txn_id']})"
        print(f"{label:<14}{note.get('policy_id', '-'):<8}{r['attempts']:<10}"
              f"{str(r['failures']):<12}{r['status']:<8}{yn(r['safe_default']):<14}"
              f"{note.get('recommended_action', '-'):<18}{yn(r['guardrail_override'])}")
    print(f"\ntotals: {calls} model calls; {first} valid on first call; {retried} recovered by retry; "
          f"{safe} safe default; {overrides} guardrail override")

    print("\n=== Extra checks ===")
    try:
        run_case("T900099", 0.9, 1000, "card payment above INR 50000 from a new device")
        print("T900099: no error (WRONG)")
    except ConnectionError as e:
        print("T900099 raises ConnectionError:", e)
    r = run_case("T800000", 0.9, 1000, "what is the cafeteria menu today")
    print("no-match question:", r["status"], "attempts", r["attempts"], "case_note", r["case_note"])
    print("validate gates:",
          validate("not json", "P1")[1],
          validate('{"summary": "x"}', "P1")[1],
          validate(json.dumps(safe_default("P2")), "P1")[1],
          validate(json.dumps(safe_default("P1")), "P1")[1])


if __name__ == "__main__":
    main()
