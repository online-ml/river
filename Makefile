COMMIT_HASH := $(shell eval git rev-parse HEAD)

download-datasets:
	python -c "from river import datasets, bandit; datasets.Elec2().download(); datasets.SMSSpam().download(); datasets.CreditCard().download(); datasets.Music().download(); datasets.CriteoAds().download(); bandit.datasets.NewsArticles().download()"

format:
	pre-commit run --all-files

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/introduction/*/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/recipes/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*.ipynb --ExecutePreprocessor.timeout=-1
	jupyter nbconvert --execute --to notebook --inplace docs/examples/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:
	jupyter nbconvert --to markdown --template docs/parse/nbconvert_template docs/introduction/*/*.ipynb
	jupyter nbconvert --to markdown --template docs/parse/nbconvert_template docs/recipes/*.ipynb
	jupyter nbconvert --to markdown --template docs/parse/nbconvert_template docs/examples/*.ipynb
	jupyter nbconvert --to markdown --template docs/parse/nbconvert_template docs/examples/*/*.ipynb

doc: render-notebooks
	python docs/parse river --out docs --verbose
	zensical build

livedoc: doc
	zensical serve

rebase:
	git fetch && git rebase origin/main

fomo:
	git fetch && git rebase origin/main

benchmark:
	uv run pytest benchmarks/codspeed --codspeed -o addopts="" $(if $(K),-k "$(K)")

# Rust criterion benches. The bench binaries link libpython (PyO3 without
# extension-module), hence the interpreter/lib-path plumbing. Setting both
# DYLD_LIBRARY_PATH (macOS) and LD_LIBRARY_PATH (Linux) is harmless.
benchmark-rust:
	PYBIN=$$(uv run --no-sync python -c "import sys; print(sys.executable)") && \
	PYLIB=$$($$PYBIN -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") && \
	PYO3_PYTHON=$$PYBIN DYLD_LIBRARY_PATH=$$PYLIB LD_LIBRARY_PATH=$$PYLIB \
	cargo bench $(if $(BENCH),--bench $(BENCH))
