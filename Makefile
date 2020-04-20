cython:
	python setup.py build_ext --inplace --force

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb --ExecutePreprocessor.timeout=-1

user-guide:
	jupyter nbconvert --to markdown docs/notebooks/*.ipynb --output-dir docs/user-guide

api-reference:
	python docs/scripts/index_api.py

doc: user-guide api-reference
	#python docs/scripts/prepare_docs.py
	mkdocs build --site-dir docs/build

livedoc: user-guide api-reference
	mkdocs serve
