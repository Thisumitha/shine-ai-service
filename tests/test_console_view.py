from evaluation import console
from evaluation.run_eval import run_evaluation


def test_question_result_shows_retrieved_vs_expected_and_scores():
    text = console.question_result(
        {"is_answerable": True, "is_refusal": False, "hit_at_1": 1.0, "hit_at_5": 1.0,
         "retrieved_pages": [5, 6], "expected_pages": [5], "answer": "A short answer.",
         "context_relevance": 0.81234, "answer_relevance": 0.7, "groundedness": 1.0, "context_recall": None},
        seconds=12.4,
    )
    assert "p.5 p.6" in text and "expected: p.5" in text and "Hit@1 yes" in text
    assert "context 0.81" in text and "grounded 1.00" in text and "12 s" in text


def test_false_refusal_and_wrong_answer_are_called_out():
    base = {"hit_at_1": 0.0, "hit_at_5": 0.0, "retrieved_pages": [], "expected_pages": [], "answer": "x"}
    assert "FALSE REFUSAL" in console.question_result({**base, "is_answerable": True, "is_refusal": True}, 1)
    assert "should have refused" in console.question_result({**base, "is_answerable": False, "is_refusal": False}, 1)
    assert "refused correctly" in console.question_result({**base, "is_answerable": False, "is_refusal": True}, 1)


def test_long_text_is_shortened_to_the_screen_width(capsys):
    report = run_evaluation(dataset_path="evaluation/sample_dataset.jsonl", mock=True, limit=2)
    out = console.summary(report, mock=True)
    assert "SYNTHETIC" in out and all(len(line) <= console.WIDTH + 2 for line in out.splitlines())
    assert "[1/2]" in capsys.readouterr().out  # progress was printed while running
