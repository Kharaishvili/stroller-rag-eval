from pathlib import Path

from stroller_rag_eval.evaluation.dataset import EvalExample
from stroller_rag_eval.evaluation.dataset import load_eval_examples
from stroller_rag_eval.evaluation.deterministic_eval import evaluate_records_deterministically
from stroller_rag_eval.evaluation.ragas_eval import records_to_ragas_rows
from stroller_rag_eval.evaluation.runner import RagEvalRecord, metadata_filter_for_example


def test_load_eval_examples_reads_semicolon_lists(tmp_path):
    dataset_path = tmp_path / "qa.csv"
    dataset_path.write_text(
        "id,manual,question,ground_truth,expected_sources,tags,must_refuse\n"
        "q1,citylite,How do I lock the stroller?,Use both rear brakes.,manual.md;brakes.md,safety;wheels,false\n"
        "q2,citylite,Does it include GPS?,Not found in the manual.,manual.md,missing_info,true\n",
        encoding="utf-8",
    )

    examples = load_eval_examples(dataset_path)

    assert len(examples) == 2
    assert examples[0].example_id == "q1"
    assert examples[0].manual == "citylite"
    assert examples[0].question == "How do I lock the stroller?"
    assert examples[0].expected_sources == ("manual.md", "brakes.md")
    assert examples[0].tags == ("safety", "wheels")
    assert examples[0].must_refuse is False
    assert examples[1].must_refuse is True


def test_load_eval_examples_defaults_missing_optional_columns(tmp_path):
    dataset_path = tmp_path / "qa.csv"
    dataset_path.write_text(
        "question\n"
        "What is covered?\n",
        encoding="utf-8",
    )

    examples = load_eval_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].example_id == "row-1"
    assert examples[0].manual is None
    assert examples[0].ground_truth is None
    assert examples[0].expected_sources == ()
    assert examples[0].tags == ()
    assert examples[0].must_refuse is False


def test_records_to_ragas_rows_preserves_required_columns():
    records = [
        RagEvalRecord(
            example_id="q1",
            manual="citylite",
            question="What is the weight limit?",
            answer="The stroller supports up to 50 lb.",
            contexts=["The stroller supports a child up to 50 lb."],
            sources=[str(Path("data/stroller.md"))],
            ground_truth="The limit is 50 lb.",
            expected_sources=["data/stroller.md"],
            tags=["safety"],
            must_refuse=False,
            retrieval_filter={"file_name": "stroller.md"},
        )
    ]

    rows = records_to_ragas_rows(records)

    assert rows == [
        {
            "question": "What is the weight limit?",
            "answer": "The stroller supports up to 50 lb.",
            "contexts": ["The stroller supports a child up to 50 lb."],
            "ground_truth": "The limit is 50 lb.",
        }
    ]


def test_metadata_filter_for_example_uses_expected_source_file_names():
    example = EvalExample(
        example_id="q1",
        manual="trailpro",
        question="Does the front wheel need to be locked?",
        ground_truth="Yes.",
        expected_sources=("trailpro_jogger_manual.md#front-wheel-lock",),
        tags=("safety",),
        must_refuse=False,
    )

    assert metadata_filter_for_example(example) == {
        "file_name": "trailpro_jogger_manual.md"
    }


def test_deterministic_eval_scores_source_and_refusal_behavior():
    records = [
        RagEvalRecord(
            example_id="q1",
            manual="citylite",
            question="Does CityLite include GPS?",
            answer="Not found in the manual.",
            contexts=["No GPS details."],
            sources=["citylite_manual.md"],
            ground_truth="Not found in the manual.",
            expected_sources=["citylite_manual.md"],
            tags=["missing_info"],
            must_refuse=True,
            retrieval_filter={"file_name": "citylite_manual.md"},
        ),
        RagEvalRecord(
            example_id="q2",
            manual="citylite",
            question="What is the child weight limit?",
            answer="I do not know.",
            contexts=["The child weight limit is 45 lb."],
            sources=["citylite_manual.md"],
            ground_truth="45 lb",
            expected_sources=["citylite_manual.md"],
            tags=["safety"],
            must_refuse=False,
            retrieval_filter={"file_name": "citylite_manual.md"},
        )
    ]

    results = evaluate_records_deterministically(records)

    assert bool(results.loc[0, "retrieved_expected_source"]) is True
    assert bool(results.loc[0, "retrieved_only_expected_sources"]) is True
    assert bool(results.loc[0, "answer_refused"]) is True
    assert bool(results.loc[0, "refusal_behavior_ok"]) is True
    assert bool(results.loc[1, "answer_refused"]) is True
    assert bool(results.loc[1, "refusal_behavior_ok"]) is False
