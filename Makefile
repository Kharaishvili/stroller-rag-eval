PYTHON ?= .venv/bin/python
TOP_K ?= 8
REPORT_DIR ?= reports/eval_topk8_hybrid_full
QUESTION ?= What is the maximum child weight for CityLite?

.PHONY: install ingest ask eval eval-smoke eval-fast test

install:
	$(PYTHON) -m pip install -e ".[dev,eval]"

ingest:
	$(PYTHON) scripts/ingest_docs.py

ask:
	$(PYTHON) scripts/run_rag.py "$(QUESTION)" --top-k $(TOP_K)

eval:
	$(PYTHON) scripts/run_eval_suite.py --top-k $(TOP_K) --report-dir $(REPORT_DIR)

eval-smoke:
	$(PYTHON) scripts/run_eval_suite.py --limit 10 --top-k $(TOP_K) --report-dir reports/eval_smoke

eval-fast:
	$(PYTHON) scripts/run_eval_suite.py --top-k $(TOP_K) --skip-ragas --skip-deepeval --report-dir reports/eval_fast

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest
