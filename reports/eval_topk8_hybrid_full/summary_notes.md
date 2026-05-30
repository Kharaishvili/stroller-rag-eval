# Evaluation Summary: Top-K 8 Hybrid Retrieval

## Run Configuration

- Dataset: `data/eval/stroller_qa.csv`
- Examples evaluated: 110
- Retrieval: Chroma vector search plus keyword fallback
- `top_k`: 8
- Judge scope: answerable rows only
- Answerable examples: 101
- Refusal examples: 9

## Why Rows Are Split

The suite evaluates two different behaviors:

- Answerable QA: questions where the manual contains the answer.
- Refusal behavior: questions where the manual does not contain the answer.

Deterministic checks run on all rows because they verify source routing and refusal behavior. RAGAS and DeepEval run only on answerable rows because unsupported/refusal questions are not normal QA examples.

## Results

| Evaluator | Metric | Score |
| --- | --- | ---: |
| Deterministic | Expected source retrieval pass rate | 100.0% |
| Deterministic | Source filter purity pass rate | 100.0% |
| Deterministic | Refusal behavior pass rate | 100.0% |
| RAGAS | Answer relevancy | 0.9120 |
| RAGAS | Context precision | 0.9070 |
| RAGAS | Faithfulness | 0.9766 |
| RAGAS | Context recall | 0.9026 |
| DeepEval | Correctness average score | 0.7744 |
| DeepEval | Correctness pass rate | 82.2% |

## Interpretation

The deterministic checks show that manual-level routing, metadata filtering, and refusal behavior are working correctly across the full dataset.

RAGAS scores indicate strong retrieval-grounded answer quality. Faithfulness is especially strong, which means generated answers are usually supported by retrieved context.

DeepEval is stricter than RAGAS on checklist and safety answers. Several DeepEval failures are semantically correct but shorter than the expected answer, or omit extra safety conditions that the expected answer includes.

## DeepEval Failure Review

DeepEval is the weakest headline metric: 82.2% pass rate with a 0.7744 average
score. That is expected for this baseline because DeepEval is acting as a strict
answer-completeness reviewer rather than a hard retrieval or refusal contract.
The deterministic checks still pass at 100.0%, and RAGAS faithfulness is 0.9766.

The row-level DeepEval run had 18 failures out of 101 answerable examples. Of
those failures, 16 scored between 0.60 and 0.70, just below the 0.70 threshold.
Manual review showed three main patterns:

- Threshold and judge artifacts: `trailpro_023` answered the full pre-jogging
  checklist, but DeepEval still scored it 0.4016 despite matching the expected
  requirements closely.
- Concise but incomplete safety answers: `citylite_018` correctly refused sand
  and gravel use, but the expected answer also listed mud, snow, rocky paths, and
  uneven trails.
- Real completeness gaps: `trailpro_032` answered that jogging with a
  9-month-old is allowed if the front wheel is locked, but it should also mention
  the other jogging requirements: weight limit, harness, smooth paved path, and
  wrist strap.

Next improvements:

- Tune the answer prompt to include all retrieved safety conditions for
  checklist-style and "yes, but only if" questions.
- Add a small regression set for multi-condition safety answers so concise
  answers do not pass when they omit important constraints.
- Keep DeepEval as a secondary review signal rather than the release gate; use
  deterministic checks for source routing, source purity, and refusal behavior.

## Report Files

- `summary.json`: Run metadata and aggregate metrics.
- `normalized_records.jsonl`: All RAG inputs and outputs.
- `answerable_records.jsonl`: Rows sent to RAGAS and DeepEval.
- `refusal_records.jsonl`: Rows evaluated as refusal behavior.
- `deterministic_results.csv`: Rule-based source/refusal checks.
- `ragas_results.csv`: RAGAS row-level results.
- `deepeval_results.csv`: DeepEval row-level results.

## Known Limitations

- DeepEval correctness is useful as a strict secondary signal, but it can penalize concise answers that are still semantically correct.
- Context precision can be affected by broad manual chunks that contain the right answer plus nearby unrelated safety details.
- The manuals are synthetic, so wording clarity directly affects retrieval and judge behavior.

## Baseline Conclusion

This run is the current baseline for the project. Hybrid retrieval with `top_k=8` produces fully passing deterministic checks and strong RAGAS scores across the answerable QA set.
