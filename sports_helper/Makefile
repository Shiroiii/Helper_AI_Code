PYTHON ?= python

version:
	@printf "Currently using executable: $(PYTHON)\n"
	which $(PYTHON)
	$(PYTHON) --version

test:
	pytest


requires:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

installed:
	pip list