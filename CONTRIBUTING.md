# Contributing

## Installation

First, create a virtual environment. You'll want to activate it every time you want to work on `creme`.

```sh
> python -m venv .venv
> source .venv/bin/activate
```

Finally, you may fork the `master` branch of the repository, which you can do from GitHub. Once you've forked the repository, clone it to your work station. Then, navigate to the cloned directory and install the required dependencies:

```sh
> pip install -e ".[dev]"
```

Finally, install `creme` in [development mode](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install):

```sh
> python setup.py develop
```

## Making changes

### Installation

Before starting you want to make sure you have Python 3.6 or above installed. We recommend you create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) with `conda`, as so:

```sh
conda create --name creme python=3.6 anaconda cython
conda activate creme
```

You also will need GCC to compile Cython extensions. If you don't already have it installed, you can do so via `conda`:

```sh
conda install -c gcc
```

Finally, you may fork the `dev` branch of the repository, which you can do from GitHub. Once you've done the fork, you can clone it to your work station. Once this is done navigate to the cloned directory and install the required dependencies:

```sh
pip install -e ".[dev]"
python setup.py develop
```

### Making changes

You're now ready to make some changes. How you make the changes is up to you of course. But you might some pointers as to how to test them. A nice and simple way is to create a script at the root of the `creme` directory. Let's say you name it `test.py`. In this script you can import whatever module or class you're changing/creating. You can then run this script via `python test.py` and check if everything works the way you it want to. Another good way to go is to write an example in the docstring of the class you're developing with. You can then run `pytest` to make sure that the outputs in the example are correct.

### Adding your change to the documentation

All classes and function are automatically picked up and added to the documentation. The only thing you have to do is to add an entry to the relevant file in the [`changelogs` directory](docs/changelogs).


## Style conventions

- Always use the `__all__` variable at the top of each module
- Use [Google style Python docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)
- When citing references, please use the Harvard referencing style. You can obtain a Harvard styled reference from Google Scholar.


## Building Cython extensions

```sh
make cython
```


## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
pytest
```

**Static typing**

These tests absolutely have to pass.

```sh
mypy creme
```

**Web dependent tests**

This involves tests that need an internet connection, such as those in the `datasets` module. In most cases you probably don't need to run these.

```sh
pytest -m web
```

**Notebook tests**

You don't have to worry too much about these, as we check them before each release. If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way round.

```sh
pytest --nbval-lax --current-env docs/notebooks/*.ipynb
```


## Making a pull request

Once you're happy with your changes, you can push them to your remote fork. By the way do not hesitate to make small commits rather than one big one, it makes things easier to review. You can create a pull request to `creme`'s `master` branch.

:warning: Don't make pull requests towards the `master` branch.


## Documentation

The documentation is built with [pdoc3](https://pdoc3.github.io/pdoc/).

```sh
pip install -e ".[docs]"  # installs dependencies required for building the docs
make livedoc  # builds the docs from scratch and starts a local server
```


## Adding contributors

- Install the [allcontributors CLI](https://allcontributors.org/docs/en/cli/installation)
- Run `yarn all-contributors add <GitHub username> code`
