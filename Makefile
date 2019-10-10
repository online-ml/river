update_nb:
	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb --ExecutePreprocessor.timeout=-1

doc:
	cd docs && $(MAKE) clean && rm -rf generated && python create_api_page.py && $(MAKE) html -j 4

clean:
	rm -rf **/*.c **/*.so **/*.pyc **/*/__pycache__/ build/

cython:
	python setup.py build_ext --inplace
