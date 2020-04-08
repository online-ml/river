pytest:
	python -m pytest

flake8:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

mypy:
	python mypy

doc:
	pdoc3 -c show_type_annotations=True -c show_inherited_members=True -c latex_math=True --template-dir pdocs/templates --html -o pdocs -f creme

livedoc:
	pdoc3 -c show_type_annotations=True -c show_inherited_members=True -c latex_math=True --template-dir pdocs/templates --http : creme
