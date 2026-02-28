.PHONY: demo doctor ui api test

demo:
	python -m ragfin.demo

doctor:
	python -m ragfin.doctor

ui:
	python -m ragfin.ui

api:
	python -m ragfin.api

test:
	python -m pytest -q
