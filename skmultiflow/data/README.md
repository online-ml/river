## Module
Data sets and generators

### Categories
* Data sets

### Functionality
* File Reader
* Stream generators

### Developing notes
* 19/05/2017 - CsvFileStream
* No optimization yet - time to give about 581k instances of 54 attributes and 1 label: 418 s
* -> 1196 inst/s -> 0.000836 s/inst

* 22/05/2017 - CsvFileStream / WaveformGenerator
* Removed labels from instances - time to give 581k instances of 54 attributes and 1 label: 100 s
* -> 5810 inst/s -> 0.000172 s/inst

* 23/05/2017 - RandomTreeGenerator
* Using the same instance structure as before - time to give 581k instances of 5 numAtt, 5 nomAtt and 1 label: 45 s
* -> 12911 inst/s -> 0.00007745 s/inst

### Format conventions
* An instance does not contain de data header, but the streamer does, so that the classifier or the evaluator can easily
    have access to that variable.