COMMIT_HASH := $(shell eval git rev-parse HEAD)

format:
	pre-commit run --all-files

cython:
	python setup.py build_ext --inplace --force

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown docs/getting-started/getting-started.ipynb
	jupyter nbconvert --to markdown docs/user-guide/*.ipynb --output-dir docs/user-guide
	jupyter nbconvert --to markdown docs/examples/*.ipynb --output-dir docs/examples

doc: render-notebooks
	yamp river --out docs/api
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

rebase:
	git fetch && git rebase origin/main
