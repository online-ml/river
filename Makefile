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

estimator-regression:
	uv run python -m benchmarks.estimator_regression.cli run --output benchmarks/estimator_regression/metrics.head.yml $(if $(K),-k $(K),)

estimator-regression-compare:
	uv run python -m benchmarks.estimator_regression.cli compare \
	  --base benchmarks/estimator_regression/metrics.base.yml \
	  --head benchmarks/estimator_regression/metrics.head.yml \
	  --output benchmarks/estimator_regression/report.md

estimator-regression-discover:
	uv run python -m benchmarks.estimator_regression.cli discover -v

estimator-regression-audit:
	uv run python -m benchmarks.estimator_regression.cli audit
