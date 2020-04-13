pytest:
	python -m pytest

flake8:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

mypy:
	python mypy creme

cython:
	python setup.py build_ext --inplace --force -X boundscheck=True

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb --ExecutePreprocessor.timeout=-1

user-guide:
	jupyter nbconvert --to markdown docs/notebooks/*.ipynb --output-dir docs/user-guide

api-reference:
	python docs/scripts/index_api.py

docs: user-guide api-reference
