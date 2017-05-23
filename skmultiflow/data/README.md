## Module
Data sets and generators

### Categories
* Data sets

### Functionality
* File Reader
* Stream generators

### Developing notes
* 19/05/2017 - No optimization yet - time to give about 581k instances of 54 attributes and 1 label: 418 s
* -> 1196 inst/s -> 0.000836 s/inst

* 22/05/2017 - Removed labels from instances - time to give 581k instances of 54 attributes and 1 label: 100 s
* -> 5810 inst/s -> 0.000172 s/inst

### Format conventions
* An instance does not contain de data header, but the streamer does, so that the classifier or the evaluator can easily
    have access to that variable.