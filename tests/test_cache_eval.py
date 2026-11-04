import json

from app import cache
from scripts import eval_cache, make_cache_pairs


def test_same_numbers():
    assert cache.same_numbers("Sara has 3 apples and buys 5", "If Sara buys 5 apples and has 3")
    assert cache.same_numbers("It costs $1,200", "The price is 1200 dollars")
    assert not cache.same_numbers("Sara has 3 apples", "Sara has 4 apples")
    assert cache.same_numbers("Capital of France?", "What's France's capital city?")   # no numbers


def test_parse_pair():
    reply = 'Sure!\n{"paraphrase": "What is France\'s capital?", "lookalike": "What is Spain\'s capital?"}'
    assert make_cache_pairs.parse_pair(reply)["lookalike"] == "What is Spain's capital?"
    assert make_cache_pairs.parse_pair("no json here") is None
    assert make_cache_pairs.parse_pair(json.dumps({"paraphrase": "x"})) is None


def make_pairs():
    return [
        {"id": "a", "category": "math", "original": "Sara has 3 apples", "paraphrase": "Sara owns 3 apples",
         "lookalike": "Sara has 4 apples"},
        {"id": "b", "category": "open_qa", "original": "Capital of France?", "paraphrase": "France's capital?",
         "lookalike": "Capital of Spain?"},
    ]


def test_lookups_and_sweep():
    # originals a, b; paraphrases a', b'; look-alikes a'', b''
    vectors = [[1, 0, 0], [0, 1, 0],          # originals
               [1, 0.05, 0], [0.05, 1, 0],    # paraphrases: very close to their own original
               [1, 0.2, 0], [0, 1, 1]]        # look-alike a'' close to a (numbers differ), b'' far from b
    rows = eval_cache.lookups(make_pairs(), vectors)

    assert all(r["matched_own"] for r in rows)
    lookalike_a = next(r for r in rows if r["kind"] == "lookalike" and r["id"] == "a")
    assert lookalike_a["similarity"] > 0.95 and not lookalike_a["same_numbers"]

    table = {(row["threshold"], row["rule"]): row for row in eval_cache.sweep(rows)}
    at_95 = table[(0.95, "similarity only")]
    assert at_95["paraphrases_caught"] == 1.0
    assert at_95["lookalikes_served"] == 0.5            # the 4-apples question slips through
    assert table[(0.95, "similarity + same numbers")]["lookalikes_served"] == 0.0   # the number rule stops it


def test_best_threshold_respects_the_wrong_hit_limit():
    table = [
        {"threshold": 0.85, "rule": "r", "paraphrases_caught": 1.0, "lookalikes_served": 0.4},
        {"threshold": 0.90, "rule": "r", "paraphrases_caught": 0.8, "lookalikes_served": 0.0},
        {"threshold": 0.95, "rule": "r", "paraphrases_caught": 0.5, "lookalikes_served": 0.0},
    ]
    assert eval_cache.best_threshold(table, "r", 0.0)["threshold"] == 0.90