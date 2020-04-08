make pytest:
	python -m pytest

make flake8:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

make mypy:
	python mypy

make doc:
	pdoc3 -c show_type_annotations=True -c show_inherited_members=True --template-dir pdocs/templates --html -o pdocs -f creme

make livedoc:
	pdoc3 -c show_type_annotations=True -c show_inherited_members=True --template-dir pdocs/templates --http : creme
