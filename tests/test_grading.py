import random

from scripts import build_dataset, grade, review_judge


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


