import random

from scripts import build_dataset,grade,review_judge


# --- Building the dataset (Day 14) ----------------------------------------

def test_gsm8k_answer_takes_the_number_after_hashes():
    solution = "She sells 16 - 3 - 4 = 9 eggs.\n9 * 2 = 18\n#### 1,018"
    assert build_dataset.gsm8k_answer(solution) == "1018"


def test_usable_everyday_skips_questions_that_need_extra_context():
    rows = [
        {"category": "open_qa", "context": "", "instruction": "Why is the sky blue?"},
        {"category": "closed_qa", "context": "", "instruction": "Not a stand-alone category"},
        {"category": "open_qa", "context": "Some article text", "instruction": "Summarize this"},
    ]
    usable = build_dataset.usable_everyday(rows)
    assert [row["instruction"] for row in usable] == ["Why is the sky blue?"]


def test_sampling_is_repeatable():
    rows = [{"question": f"Q{i}", "answer": f"#### {i}"} for i in range(50)]
    first = build_dataset.math_questions(rows, 5, random.Random(42))
    second = build_dataset.math_questions(rows, 5, random.Random(42))
    assert first == second


def test_to_number_handles_money_and_commas():
    assert grade.to_number("$1,250") == 1250
    assert grade.to_number("30.00") == 30
    assert grade.to_number("Tokyo") is None


def test_last_number_reads_the_final_answer():
    assert grade.last_number("17 x 23: first 17 x 20 = 340, then 340 + 51 = 391") == 391
    assert grade.last_number("No numbers here") is None


def test_number_matches():
    assert grade.number_matches("So the total is 1,018 dollars.", "1018")
    assert not grade.number_matches("The answer is 1017.", "1018")


def test_grading_method():
    assert grade.grading_method({"reference": "391"}) == "number"
    assert grade.grading_method({"reference": "Tokyo"}) == "judge"
    assert grade.grading_method({"reference": None}) == "judge"
    assert grade.grading_method({"grading": "judge", "reference": "7"}) == "judge"


def test_parse_verdict():
    assert grade.parse_verdict("PASS\nThe answer is correct.") is True
    assert grade.parse_verdict("\nfail\nIt misses the key point.") is False
    assert grade.parse_verdict("**PASS**\nCorrect.") is True
    assert grade.parse_verdict("I think it is mostly fine.") is None

def test_agreement_rate():
    pairs = [(True, True), (False, False), (True, False), (True, True)]
    assert review_judge.agreement(pairs) == 0.75
    assert review_judge.agreement([]) == 0.0