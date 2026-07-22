# Contribution guidelines

## What to work on?

Take a look at our [GitHub issues](https://github.com/online-ml/river/issues). The labelling scheme should be self-explanatory. You're welcome to pick anything that takes your fancy and that you deem important. We encourage you to discuss with us your approach before you start the implementation, to avoid wasting time on out-of-scope work.

We do not assign issues to people. If you want to indicate you're working on something, just start a draft pull request, indicating the issue you're targeting.

Of course, you're welcome to propose and contribute new ideas. We encourage you to [open a discussion](https://github.com/online-ml/river/discussions/new) so that we can have a chat and align.

## Rules about coding agents

We are not against coding agents. But River was made by humans who enjoy working with each other, and we want to preserve that human touch. Here are our rules:

> 1. Coding agents can write code, but not comments.

We have a codebase that is of good quality, with enough examples for coding agents to write idiomatic code. Therefore, AI generated code is not a problem per say. But using an AI to write comments is worrying, because it's a sign we did not put in the effort to understand the generated code.

> 2. Prose is written by humans. This covers issues, pull request descriptions, commit messages, docstrings, release notes, and any kind of discussion.

We don't want coding agents to do the high-level thinking for us. Therefore, we should force ourselves to write all our discussions with our own words. AI generated prose almost always reads like slop, and too much of it is off-putting. We believe using our own words is more polite, friendly, and enjoyable for everyone. Docstrings and release notes count too: they're how we talk to our users, so they deserve the same care.

Of course, you can use a coding agent to run a benchmark and produce a summary table. But you should editorialize and insert it into a message you've written yourself.

> 3. Code written by agents should be disclosed as such.

We should not deceive each other by asking an AI to generate code, and merging it into the codebase without indicating its source. We want to be able to differentiate between the two. A `Co-authored-by:` trailer on the commit is a simple way to do this.

> 4. Be thorough on tests.

Good tests usually span more lines that implementations themselves. They can be tedious to write. Access to coding agents means there is no more excuse for not writing tests.

> 5. Align before you build.

Don't let an agent open a drive-by pull request. As above, discuss your approach with us first, and start from a draft pull request. This matters all the more when an agent makes it cheap to produce a lot of code quickly.

> 6. You are accountable for what your agent submits.

An agent acting on your behalf is still you. You own its output, and our [Code of Conduct](CODE_OF_CONDUCT.md) applies to it just as it does to anything you write yourself.

> 7. Any infringement of the rules above allows the maintainers to close any associated discussion or pull request.

*These rules are enforced in `AGENTS.md`.*

## Fork/clone/pull

The typical workflow for contributing to River is:

1. Fork the `main` branch from the [GitHub repository](https://github.com/online-ml/river/).
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

## Local setup

Start by cloning the repository:

```sh
git clone --single-branch https://github.com/online-ml/river
```

> **Note:** The `--single-branch` flag is important. Without it, Git will also fetch the `gh-pages` branch which contains the generated documentation site, adding several hundred MiB to the clone.

Next, you'll need a Python environment. A nice way to manage your Python versions is to use pyenv, which can installed [here](https://github.com/pyenv/pyenv-installer). Once you have pyenv, you can install the latest Python version River supports:

```sh
pyenv install -v $(cat .python-version)
```

You need a `Rust` compiler you can install it by following this [link](https://www.rust-lang.org/fr/tools/install). You'll also need [uv](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now you're set to install River:

```sh
uv sync
```

Finally, install the [prek](https://github.com/j178/prek) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
uv run prek install --hook-type pre-push
```

You can optionally run `prek` at any time as so:

```sh
uv run prek run --all-files
```

## Making changes

You're now ready to make some changes. We strongly recommend that you to check out River's source code for inspiration before getting into the thick of it. How you make the changes is up to you of course. However we can give you some pointers as to how to test your changes. Here is an example workflow that works for most cases:

- Create and open a Jupyter notebook at the root of the directory.
- Add the following in the code cell:

```py
%load_ext autoreload
%autoreload 2
```

- The previous code will automatically reimport River for you whenever you make changes.
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

If you're adding a class or a function, then you'll need to add a docstring. We follow the [Numpy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html), so please do too.

To build the documentation, you need to install some extra dependencies:

```sh
uv sync --group docs
```

From the root of the repository, you can then run the `make livedoc` command to take a look at the documentation in your browser. This will run a custom script which parses all the docstrings and generate MarkDown files that [MkDocs](https://www.mkdocs.org/) can render.

## Adding a release note

All classes and function are automatically picked up and added to the documentation. The only thing you have to do is to add an entry to the relevant file in the [`docs/releases` directory](docs/releases).

## Build the Rust extensions

River's Rust extensions (under `rust_src/`) are built automatically by `uv sync`:

```sh
uv sync
```

When iterating on the Rust code, rebuilding the whole project with `uv sync` is slower than necessary. Use `maturin develop` for a tighter inner loop. Note that `maturin` is not installed in the environment, so run it through `uv run --with`:

```sh
uv run --with maturin maturin develop --release
```

(Plain `uv run maturin ...` fails with "Failed to spawn: maturin" because the binary isn't on the environment's path.)

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
uv run pytest
```

**Static typing**

These tests absolutely have to pass.

```sh
uv run mypy river
```

**Web dependent tests**

This involves tests that need an internet connection, such as those in the `datasets` module which requires downloading some files. In most cases you probably don't need to run these.

```sh
uv run pytest -m web
```

### Benchmarks

Performance-sensitive changes should come with a benchmark. Run `make benchmark` locally;
CI posts exact measured results on your pull request via CodSpeed. See
[`benchmarks/README.md`](benchmarks/README.md) for the local commands, determinism rules,
and the benchmark template. New estimators that are expected to be hot paths should add a
benchmark under `benchmarks/codspeed/python/` in the same pull request.

**Notebook tests**

You don't have to worry too much about these, as we only check them before each release. If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way around.

```sh
uv run make execute-notebooks
```

## Making a new release

1. Checkout `main`
2. Run `uv run make execute-notebooks` just to be safe
3. Bump the version in `river/__version__.py`
4. Bump the version in `pyproject.toml` (then run `uv lock`)
5. Rename `docs/releases/unreleased.md` to `docs/releases/X.Y.Z.md` and add the release date to its top heading. If no `unreleased.md` exists (no changes were accumulated), create `X.Y.Z.md` directly.
6. Update the Releases nav in `mkdocs.yml`: add the new version entry at the top of the list.
7. Commit and push

> Note: `docs/releases/unreleased.md` is created on demand by contributors when the first change worth noting lands after a release. When created, it must also be added to the Releases nav in `mkdocs.yml`. Do not pre-create an empty `unreleased.md` — an empty page will 404 in the docs.
8. Wait for CI to [run the unit tests](https://github.com/online-ml/river/actions/workflows/ci.yml)
9. Push the tag:

```sh
RIVER_VERSION=$(uv run python -c "import river; print(river.__version__)")
echo $RIVER_VERSION
```

```sh
git tag $RIVER_VERSION -m "Release $RIVER_VERSION"
git push origin $RIVER_VERSION
```

10. Wait for CI to [ship to PyPI](https://github.com/online-ml/river/actions/workflows/pypi.yml)
11. Check the [new docs have been published](https://github.com/online-ml/river/actions/workflows/release-docs.yml)
12. Create a [release](https://github.com/online-ml/river/releases):

```sh
RELEASE_NOTES=$(cat <<-END
- https://riverml.xyz/${RIVER_VERSION}/releases/${RIVER_VERSION}/
- https://pypi.org/project/river/${RIVER_VERSION}/
END
)
brew update && brew install gh
gh release create $RIVER_VERSION --notes $RELEASE_NOTES
```

13. Pyodide needs to be told there is a new release. This can done by updating [`packages/river`](https://github.com/online-ml/pyodide/tree/main/packages/river) in [online-ml/pyodide](https://github.com/online-ml/pyodide)
