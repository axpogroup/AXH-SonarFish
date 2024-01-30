all: hooks

hooks:
	pre-commit install
	pre-commit install --hook-type pre-push
