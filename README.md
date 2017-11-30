![logo](docs/_static/images/skmultiflow-logo-wide.png?raw=true "Title")


A multi-output/multi-label and stream data framework.
Inspired by [MOA](https://moa.cms.waikato.ac.nz/) and [MEKA](http://meka.sourceforge.net/),
 following [scikit-learn](http://scikit-learn.org/stable/) philosophy.

* [Webpage](https://scikit-multiflow.github.io/)
* [Documentation](https://scikit-multiflow.github.io/scikit-multiflow/)
* Google Group: TODO

### Project leaders

* Jacob MONTIEL
* Jesse READ
* Albert BIFET

### Contributors

* Guilherme KURIKE MATSUMOTO

### matplotlib backend considerations
* You may need to change your matplotlib backend, because not all backends work
in all machines.
* If this is the case you need to check
[matplotlib's configuration](https://matplotlib.org/users/customizing.html).
In the matplotlibrc file you will need to change the line:  
    ```
    backend     : Qt5Agg  
    ```
    to:  
    ```
    backend     : another backend that works on your machine
    ```  
* The Qt5Agg backend should work with most machines, but a change may be needed.

### License
* 3-Clause BSD License
