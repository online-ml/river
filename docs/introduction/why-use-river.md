# Why use River?

## Processing one sample at a time

All the tools in the library can be updated with a single observation at a time. They can therefore be used to process streaming data. Depending on your use case, this might be more convenient than using a batch model.

## Adapting to drift

In the streaming setting, data can evolve. Adaptive methods are specifically designed to be robust against concept drift in dynamic environments. Many of River's models can cope with concept drift.

## General purpose

River supports different machine learning tasks, including regression, classification, and unsupervised learning. It can also be used for adhoc tasks, such as computing online metrics, as well as concept drift detection.

## User experience

River is not the only library allowing you to do online machine learning. But it might just the simplest one to use in the Python ecosystem. River plays nicely with Python dictionaries, therefore making it easy to use in the context of web applications where JSON payloads are aplenty.
