PYTHON ?= python3

.PHONY: test run

test:
	$(PYTHON) -m unittest discover -s tests -v

run:
	$(PYTHON) server.py --host 127.0.0.1 --port 8000
