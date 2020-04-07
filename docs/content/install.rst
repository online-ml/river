Installation
============

``creme`` is intended to work with Python 3.6 or above.

Installation can be done by using ``pip``:

.. code-block:: bash

    pip install creme

`creme` also has some extra dependencies that not installed by default. This includes utilities for connecting with other libraries, which includes pandas, PyTorch, scikit-learn, and SQLAlchemy. You have the liberty to install these dependencies separately by yourself. You can also install of all of them with the following command:

.. code-block:: bash

    pip install "creme[compat]"

There are `wheels available <https://pypi.org/project/creme/#files>`_ for Linux, MacOS, and Windows. You can also install the latest development version as so:

.. code-block:: bash

    pip install git+https://github.com/creme-ml/creme

    # Or, through SSH:
    pip install git+ssh://git@github.com/creme-ml/creme.git

Note that installing the development version requires already having `Cython <https://github.com/cython/cython>`_ installed.
