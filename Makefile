PY=.venv/bin/python
PY39=.venv39/bin/python

.PHONY: test test-py test-ext test-int

test: test-py test-ext

lint-py:
	cd python && ../.venv/bin/ruff check .

test-py: lint-py
	cd python && ../$(PY) -m pytest -q
	cd python && ../$(PY39) -m pytest -q

test-ext:
	cd extension && npm run -s typecheck && npm run -s test:unit

test-int:
	cd extension && npm run -s test:int
