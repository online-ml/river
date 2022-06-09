COMMIT_HASH := $(shell eval git rev-parse HEAD)

format:
	pre-commit run --all-files

cython:
	python setup.py build_ext --inplace --force

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown docs/introduction/*/*.ipynb
	jupyter nbconvert --to markdown docs/recipes/**.ipynb
	jupyter nbconvert --to markdown docs/examples/**.ipynb

doc: render-notebooks
	(cd benchmarks && python render.py)
	yamp river --out docs/api --verbose
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

rebase:
	git fetch && git rebase origin/main
