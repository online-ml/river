update_nb:
	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb --ExecutePreprocessor.timeout=-1

clean:
	rm -f **/*.c **/*.so **/*.pyc
	rm -rf **/*/__pycache__ build .ipynb_checkpoints docs/notebooks/.ipynb_checkpoints .pytest_cache .empty .eggs creme.egg-info dist

cython:
	python setup.py build_ext --inplace --force

doc:
	cd docs && $(MAKE) clean && rm -rf content/generated && python scripts/make_api.py && $(MAKE) html -j 4

livedoc:
	sphinx-autobuild docs docs/_build/html --port 0 --open-browser --delay 0
