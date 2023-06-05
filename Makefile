COMMIT_HASH := $(shell eval git rev-parse HEAD)

format:
	pre-commit run --all-files

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/introduction/*/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/recipes/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown docs/introduction/*/*.ipynb
	jupyter nbconvert --to markdown docs/recipes/*.ipynb
	jupyter nbconvert --to markdown docs/examples/*.ipynb
	jupyter nbconvert --to markdown docs/examples/*/*.ipynb

doc: render-notebooks
	(cd benchmarks && python render.py)
	python docs/parse river --out docs --verbose
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

rebase:
	git fetch && git rebase origin/main

develop:
	python ./setup.py develop

build-cython:
	python setup.py build_ext --inplace --force

build-rust:
	python setup.py build_rust --inplace --release

build:
	python setup.py build_rust --inplace --release build_ext  --inplace --force
