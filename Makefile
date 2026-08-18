.PHONY: api test frontend-build

api:
	python -m ragfin.api

test:
	python -m pytest -q

frontend-build:
	cd frontend && npm run build
