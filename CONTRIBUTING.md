# Contributing

## Advised development cycle

### Installation

Before starting you want to make sure you have Python 3.6 or above installed.

You first want to fork the `dev` branch of the repository, which you can do from GitHub. Once you've done the fork, you can clone it to your work station. Once this is done navigate to the cloned directory and install the required dependencies:

```sh
python setup.py develop
pip install -e ".[dev]"
```

:point_up: We recommend that you use [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

:point_up: We also recommend that you to use a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

### Making changes

You're now ready to make some changes. How you make the changes is up to you of course. But you might some pointers as to how to test them. A nice and simple way is to create a script at the root of the `creme` directory. Let's say you name it `test.py`. In this script you can import whatever module or class you're changing/creating. You can then run this script via `python test.py` and check if everything works the way you it want to. Another good way to go is to write an example in the docstring of the class you're developing with. You can then run `pytest` to make sure that the outputs in the example are correct.

### Adding your change to the documentation

If you've added a new functionality, then you will have to write a docstring and add it to the documentation. To do so you can edit the `docs/api.rst` file accordingly. Feel free to ask for help. If you've made a change to an existing class then you have to make sure that you've updated it's docstring accordingly. To make sure your modifications to the documentation are okay go to the `docs` directory and run `make html`.


## Style conventions

- Always use the `__all__` variable at the top of each module
- Use [Google style Python docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)


## Making a pull request

Once you're happy with your changes, you can push them to your remote fork. By the way do not hesitate to make small commits rather than one big one, it makes things easier to review. You can create a pull request to `creme`'s `master` branch.

:warning: Don't make pull requests towards the `master` branch.


## Documentation

The documentation is built with [Sphinx](http://www.sphinx-doc.org/en/master/).

```sh
cd /path/to/creme/docs/
make html
```
