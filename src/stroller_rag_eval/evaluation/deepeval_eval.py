"""DeepEval adapter for normalized RAG evaluation records"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any, cast

from stroller_rag_eval.evaluation.runner import RagEvalRecord, write_records_jsonl


DEFAULT_METRIC_NAMES = ("correctness",)
SUPPORTED_METRIC_NAMES = (
    "correctness",
    "answer_relevancy",
    "faithfulness",
    "contextual_relevancy",
)


def evaluate_records_with_deepeval(
    records: list[RagEvalRecord],
    *,
    model: str,
    threshold: float,
    output_path: Path | None = None,
    async_mode: bool = False,
    include_reason: bool = False,
    show_indicator: bool = False,
    max_concurrent: int = 1,
    truths_extraction_limit: int | None = 5,
    metric_names: tuple[str, ...] = DEFAULT_METRIC_NAMES,
    provider_timeout: float = 60,
    task_timeout: float = 180,
) -> Any:
    """Run DeepEval RAG metrics over normalized RAG evaluation records"""

    _set_deepeval_env_defaults(
        provider_timeout=provider_timeout,
        task_timeout=task_timeout,
    )

    try:
        from deepeval import evaluate
        from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.models import GPTModel
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise RuntimeError(
            "DeepEval dependencies are not installed. "
            'Run `pip install -e ".[eval]"` first.'
        ) from exc

    evaluator = GPTModel(model=model, timeout=provider_timeout)
    metrics = []
    for metric_name in metric_names:
        if metric_name == "correctness":
            metrics.append(
                GEval(
                    name="Correctness",
                    evaluation_params=[
                        SingleTurnParams.INPUT,
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT,
                    ],
                    criteria=(
                        "Score whether the actual output correctly answers the input "
                        "according to the expected output. Treat semantically equivalent "
                        "wording as correct. For safety yes/no questions, a direct refusal "
                        "or prohibition is correct when it matches the expected output. "
                        "Penalize contradictions, missing required limits, or answers about "
                        "the wrong product."
                    ),
                    threshold=threshold,
                    model=evaluator,
                    async_mode=async_mode,
                )
            )
        elif metric_name == "answer_relevancy":
            metrics.append(
                AnswerRelevancyMetric(
                    threshold=threshold,
                    model=evaluator,
                    include_reason=include_reason,
                    async_mode=async_mode,
                )
            )
        elif metric_name == "faithfulness":
            metrics.append(
                FaithfulnessMetric(
                    threshold=threshold,
                    model=evaluator,
                    include_reason=include_reason,
                    async_mode=async_mode,
                    truths_extraction_limit=truths_extraction_limit,
                )
            )
        elif metric_name == "contextual_relevancy":
            metrics.append(
                ContextualRelevancyMetric(
                    threshold=threshold,
                    model=evaluator,
                    include_reason=include_reason,
                    async_mode=async_mode,
                )
            )
        else:
            supported_names = ", ".join(SUPPORTED_METRIC_NAMES)
            raise ValueError(
                f"Unsupported DeepEval metric '{metric_name}'. "
                f"Choose from: {supported_names}."
            )

    test_cases = [
        LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ground_truth,
            retrieval_context=cast(list[Any], list(record.contexts)),
        )
        for record in records
    ]

    if output_path is not None:
        write_records_jsonl(records, output_path)

    stdout_context = (
        contextlib.nullcontext()
        if show_indicator
        else contextlib.redirect_stdout(io.StringIO())
    )
    with stdout_context:
        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=async_mode,
                max_concurrent=max_concurrent,
            ),
            display_config=DisplayConfig(
                show_indicator=show_indicator,
                print_results=False,
                inspect_after_run=False,
            ),
            error_config=ErrorConfig(ignore_errors=False),
        )

    return result


def _set_deepeval_env_defaults(
    *,
    provider_timeout: float,
    task_timeout: float,
) -> None:
    """Keep DeepEval smoke runs from waiting on long default retries"""

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")
    os.environ.setdefault(
        "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
        _format_seconds(provider_timeout),
    )
    os.environ.setdefault(
        "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
        _format_seconds(task_timeout),
    )


def _format_seconds(value: float) -> str:
    if value <= 0:
        raise ValueError("DeepEval timeout values must be greater than 0.")
    return str(float(value))
