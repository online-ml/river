COMMIT_HASH := $(shell eval git rev-parse HEAD)

format:
	pre-commit run --all-files

cython:
	python setup.py build_ext  --inplace --force

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

clean_rust_setup:
	rm river/stats/_rust_stats.cpython*
	rm -rf target
	rm -rf river.egg-info
	rm Cargo.lock
	rm -rf build

rust_develop:
	python ./setup.py develop

rust_release:
	python setup.py build_rust --inplace --release

build_all:
	python setup.py build_rust --inplace --release build_ext  --inplace --force