PYTHON ?= python

version:
	@printf "Currently using executable: $(PYTHON)\n"
	which $(PYTHON)
	$(PYTHON) --version

test:
	pytest

virtual:
	$(PYTHON) -m venv helper

requires:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

installed:
	pip list