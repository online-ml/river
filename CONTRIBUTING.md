# Contribution guidelines

## Installation

First, create a virtual environment. You'll want to activate it every time you want to work on `creme`.

```sh
> python -m venv .venv
> source .venv/bin/activate
```

You can also use a `conda` environment, as explained [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/).

You then want to fork the `master` branch of the repository, which you can do from GitHub's interface. Once you've forked the repository, clone it to your work station. Then, navigate to the cloned directory and install the required dependencies:

```sh
> pip install -e ".[dev]"
```

Finally, install `creme` in [development mode](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install):

```sh
> python setup.py develop
```

## Making changes

You're now ready to make some changes. How you make the changes is up to you of course. However we can give you some pointers as to how to test your changes. A nice and simple way is to create a script at the root of the `creme` directory. Let's say you name it `foo.py`. In this script you can import whatever module or class you're changing/creating. You can then run this script via `python foo.py` and check if everything works the way you it want to. Another good way to go is to write an example in the docstring of the class you're developing with. You can then run `pytest path/to/foo.py` to make sure that the outputs in the example are correct.

We strongly invite you to check out `creme`'s source code for inspiration.

## Documenting your change

If you're adding a class or a function, then you'll need to add a docstring. We follow the [Google docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), so please do too.

To build the documentation, you need to install some extra dependencies:

```sh
> pip install -e ".[docs]"
```

From the root of the repository, you can then run the `make livedoc` command to take a look at the documentation in your browser. This will run a custom script which parses all the docstrings and generate MarkDown files that [MkDocs](https://www.mkdocs.org/) can render.

## Adding a release note

All classes and function are automatically picked up and added to the documentation. The only thing you have to do is to add an entry to the relevant file in the [`docs/releases` directory](docs/releases).

## Building Cython extensions

```sh
> make cython
```

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
> pytest
```

**Static typing**

These tests absolutely have to pass.

```sh
> mypy creme
```

**Web dependent tests**

This involves tests that need an internet connection, such as those in the `datasets` module which requires downloading some files. In most cases you probably don't need to run these.

```sh
> pytest -m web
```

**Notebook tests**

You don't have to worry too much about these, as we check them before each release. If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way round.

```sh
> pytest --nbval-lax --current-env docs/notebooks/*.ipynb
```

## Making a pull request

Once you're happy with your changes, you can push them to your remote fork. By the way do not hesitate to make small commits rather than one big one, it makes things easier to review. You can create a pull request to `creme`'s `master` branch.

## Adding contributors

- Install the [allcontributors CLI](https://allcontributors.org/docs/en/cli/installation)
- Run `yarn all-contributors add <GitHub username> code`
