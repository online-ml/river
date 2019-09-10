# CONTRIBUTING

We welcome contributions from the community. Here you will find information to start contributing to `scikit-multiflow`.

## Contribution to a GitHub project
If you are not familiar with Git or GitHub, here are good resources to start in the right direction:
* [The beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/)
* [How to make a clean pull request](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## Code style and documentation

* Python Code shall comply with [PEP 8](https://www.python.org/dev/peps/pep-0008/)

* Documentation shall be in docstring format and shall follow the
  [NumPy/SciPy guidelines](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)

  - An example from the sphinx documentation:  
    https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html

## Development
This section contains relevant information for development, including project setup, coding best practices, etc.
 
### pip setup in editable mode
As default, `pip` installs a *fixed* version of a python package. However, during development, new code is added to a project incrementally and it is desired for developers to "see" these changes reflected immediately. For this purpose, `pip install` provides an `editable mode` option, to avoid re-running the setup script as new code is added.

To setup`scikit-multiflow` in editable mode you must run one of the following commands from the local path of the project:
```shell
pip install -e .
```
```shell
pip install --editable .
```

### Random number generators
Random number generators are handled as in `scikit-learn`. Meaning that, random number generators/seeds are refered as `random_state`. Before using a `random_state` object, we must ensure that it is valid, this is done via the utility function `skmultiflow.core.utils.validation.check_random_state`.

## Testing
### Run tests
We use [pytest](https://docs.pytest.org/) to maintain the quality of the framework. Always run the tests before and after your changes to ensure that everything is working as expected. To run the tests, from the package's root directory:
```bash
python setup.py test
```
or
```bash
pytest tests/some_module/some_file.py --showlocals -v
```


### Write/update tests
If you are adding new code:
* It is your responsibility to ensure that the code is correct and maintainable (by people other than you). The development team can provide support during the code review.
* Evidence of **correctness** of the results must be provided, this includes (but it is not limited to) plots, benchmarks, etc.
* Functional test(s) must be included as part of the Pull Request. These tests shall focus on providing coverage and ensuring the integrity of the project. They are intended to catch **unintentional** changes that could be introduced by unrelated development efforts.
  * We use [codecov](https://codecov.io/gh/scikit-multiflow/scikit-multiflow) to check for coverage and the corresponding report is automatically generated (updated) as part of the Pull Request.
  * You can generate the coverage report locally to ensure that the tests are reaching (activating) most of the code. For this you need the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin
    ```bash
    pytest --cov=src/skmultiflow tests/some_module/some_file.py --showlocals -v
    ```

If you are modifying existing code:
* Same rules apply regarding correctness ad testing of code. However, tests might only require to be updated.

## Sphinx documentation
* We generate our documentation using `sphinx` with the following dependencies: `sphinx_rtd_theme` 
* To update the documentation, perform in a terminal the steps below:
    * Go to scikit-multiflow/docs and type in:  
      ``` bash
      $ make html
      ```
      This will generate the documentation page in `docs/_build/html`
     
    * The documentation page is hosted in the **gh-pages** branch.

* When adding/modifying documentation, it is recommended to generate the html page locally to ensure that it is correctly generated and the content is rendered as expected.