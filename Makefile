cython:
	python setup.py build_ext --inplace --force

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*.ipynb --ExecutePreprocessor.timeout=-1

user-guide:
	jupyter nbconvert --to markdown docs/user-guide/*.ipynb --output-dir docs/user-guide
	jupyter nbconvert --to markdown docs/examples/*.ipynb --output-dir docs/examples

api-reference:
	python docs/scripts/index_api.py

doc: user-guide api-reference
	#python docs/scripts/prepare_docs.py
	mkdocs build --site-dir docs/build

livedoc: user-guide api-reference
	mkdocs serve
