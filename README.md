# scikit-multiflow

A multi-output and stream data framework.

For more information, visit our webpage: https://scikit-multiflow.github.io/scikit-multiflow/

### Project leaders

* Jacob MONTIEL
* Jesse READ
* Albert BIFET

### Contributors

* Guilherme KURIKE MATSUMOTO

### matplotlib backend considerations
* You may need to change your matplotlib backend, because not all backends work in all machines
* If this is the case you can find the conda virtual environment matplotlib config file in:
    /miniconda3/pkgs/matplotlib-2.0.2-np112py35_0/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc
* In the matplotlibrc file you will need to change the line:
    backend     : Qt5Agg
    to:
    backend     : another backend that works on your machine
* The Qt5Agg backend should work with most machines, but a change may be needed.

### License
* 3-Clause BSD License
