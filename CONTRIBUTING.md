# Contribution guidelines

## Fork/clone/pull

The typical workflow for contributing to `river` is:

1. Fork the `master` branch from the [GitHub repository](https://github.com/online-ml/river/).
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `master` branch.

## Local setup

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on `river`.

```sh
$ python -m venv .venv
$ source .venv/bin/activate
```

You can also create a virtual environment via `conda`:

```sh
$ conda create -n river -y python
$ conda activate river
```

Then, navigate to your cloned fork and install the required dependencies:

```sh
$ pip install -e ".[dev]"
```

Next, install `river` in [development mode](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install):

```sh
$ python setup.py develop
```

Finally, install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
$ pre-commit install --hook-type pre-push
```

## Making changes

You're now ready to make some changes. We strongly recommend that you to check out `river`'s source code for inspiration before getting into the thick of it. How you make the changes is up to you of course. However we can give you some pointers as to how to test your changes. Here is an example workflow that works for most cases:

- Create and open a Jupyter notebook at the root of the directory.
- Add the following in the code cell:
```py
%load_ext autoreload
%autoreload 2
```
- The previous code will automatically reimport `river` for you whenever you make changes.
- For instance, if a change is made to `linear_model.LinearRegression`, then rerunning the following code doesn't require rebooting the notebook:
```py
from river import linear_model

model = linear_model.LinearRegression()
```

## Creating a new estimator

1. Pick a base class from the `base` module.
2. Check if any of the mixin classes from the `base` module apply to your implementation.
3. Make you've implemented the required methods, with the following exceptions:
   1. Stateless transformers do not require a `learn_one` method.
   2. In case of a classifier, the `predict_one` is implemented by default, but can be overridden.
4. Add type hints to the parameters of the `__init__` method.
5. If possible provide a default value for each parameter. If, for whatever reason, no good default exists, then implement the `_unit_test_params` method. This is a private method that is meant to be used for testing.
6. Write a comprehensive docstring with example usage. Try to have empathy for new users when you do this.
7. Check that the class you have implemented is imported in the `__init__.py` file of the module it belongs to.
8. When you're done, run the `utils.check_estimator` function on your class and check that no exceptions are raised.

## Documenting your change

If you're adding a class or a function, then you'll need to add a docstring. We follow the [Google docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), so please do too.

To build the documentation, you need to install some extra dependencies:

```sh
$ pip install -e ".[docs]"
```

From the root of the repository, you can then run the `make livedoc` command to take a look at the documentation in your browser. This will run a custom script which parses all the docstrings and generate MarkDown files that [MkDocs](https://www.mkdocs.org/) can render.

## Adding a release note

All classes and function are automatically picked up and added to the documentation. The only thing you have to do is to add an entry to the relevant file in the [`docs/releases` directory](docs/releases).

## Building Cython extensions

```sh
$ make cython
```

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
$ pytest
```

**Static typing**

These tests absolutely have to pass.

```sh
$ mypy river
```

**Web dependent tests**

This involves tests that need an internet connection, such as those in the `datasets` module which requires downloading some files. In most cases you probably don't need to run these.

```sh
$ pytest -m web
```

**Notebook tests**

You don't have to worry too much about these, as we only check them before each release. If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way around.

```sh
$ make execute-notebooks
```
