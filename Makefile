update_nb:
	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb --ExecutePreprocessor.timeout=-1

doc:
	cd docs && \
		$(MAKE) clean && \
		rm -rf generated && \
		python make_datasets_table.py && \
		python make_api_page.py && \
		$(MAKE) html -j 4

clean:
	rm -f **/*.c **/*.so **/*.pyc
	rm -rf **/*/__pycache__ build .ipynb_checkpoints .pytest_cache .empty .eggs creme.egg-info dist

cython:
	python setup.py build_ext --inplace --force
