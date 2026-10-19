"""retrieval.py - Part C: TF-IDF retrieval of the fraud policy that matches a question.

    from retrieval import retrieve, MIN_SCORE
    python retrieval.py    # prints the retrieval table
"""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Highest out-of-scope score is 0.4251, lowest in-scope score is 0.4862;
# 0.45 sits between them with room on both sides.
MIN_SCORE = 0.45

with open("fraud_policies.json") as f:
    POLICIES = json.load(f)

_vectorizer = TfidfVectorizer()                       # default settings, no stop-word removal
_matrix = _vectorizer.fit_transform([p["text"] for p in POLICIES])


def retrieve(query):
    """Return (policy_id, policy_text, score) of the best-matching policy."""
    scores = cosine_similarity(_vectorizer.transform([query]), _matrix)[0]
    i = int(scores.argmax())
    return POLICIES[i]["policy_id"], POLICIES[i]["text"], float(scores[i])


def main():
    with open("retrieval_queries.json") as f:
        queries = json.load(f)
    print(f"{'query':<66}{'expected':>9}{'retrieved':>10}{'score':>8}{'answered':>10}")
    hits = in_scope = 0
    in_scores, out_scores = [], []
    for q in queries:
        pid, _, score = retrieve(q["query"])
        exp = q["expected_policy"]
        answered = score >= MIN_SCORE
        print(f"{q['query']:<66}{str(exp):>9}{pid:>10}{score:>8.4f}{str(answered):>10}")
        if exp is None:
            out_scores.append(score)
        else:
            in_scope += 1
            hits += pid == exp
            in_scores.append(score)
    print(f"\nin-scope top-1 hits: {hits} of {in_scope}")
    print(f"highest out-of-scope score {max(out_scores):.4f} < MIN_SCORE {MIN_SCORE} "
          f"<= lowest in-scope score {min(in_scores):.4f}")


if __name__ == "__main__":
    main()
