# CONTRIBUTING

We welcome contributions from the community. Here you will find the minimum information to start contributing to `scikit-multiflow`.

## Contribution to a GitHub
If you are not familiar with Git or GitHub, here are good resources to start in the right direction:
* [The beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/)
* [How to make a clean pull request](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## Code style and documentation

* Python Code shall comply with [PEP 8](https://www.python.org/dev/peps/pep-0008/)

* Documentation shall be in docstring format and shall follow the
  [NumPy/SciPy guidelines](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)

  - An example from the sphinx documentation:  
    http://www.sphinx-doc.org/en/stable/ext/example_numpy.html

## Testing
### Run tests
We use [pytest](https://docs.pytest.org/) to maintain the quality of the framework. Always run the tests to ensure that everything is working as expected. To run the tests, from the package's root directory:
```bash
python setup.py test
```

### Write tests
If you are adding new code to `scikit-multiflow` then it is highly encouraged that you include a test file as part of your Pull Request.


## Sphinx documentation
* We generate our documentation using `sphinx` with the following dependencies: `sphinx_rtd_theme` 
* To update the documentation, perform in a terminal the steps below:
    * Make your way to the scikit-multiflow parent directory (outside the project)
    * Type in:  
      ``` bash
      $ sphinx-apidoc -o scikit-multiflow/docs scikit-multiflow/ -e -f
      ```

    * Go to scikit-multiflow/docs and type in:  
      ``` bash
      $ make html
      ```
     
    * The documentation page is generated using the **gh-pages** branch.
