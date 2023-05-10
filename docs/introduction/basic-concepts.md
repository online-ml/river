# Basic concepts

Here are some concepts to give you a feel for what problems River addresses.

## Data streams

River is a library to build online machine learning models. Such models operate on data streams. But a data stream is a bit of a vague concept.

In general, a data stream is a sequence of individual elements. In the case of machine learning, each element is a bunch of features. We call these samples, or observations. Each sample might follow a fixed structure and always contain the same features. But features can also appear and disappear over time. That depends on the use case.

## Reactive and proactive data streams

The origin of a data stream can vary, and usually it doesn't matter. You should be able to use River regardless of where your data comes from. It is however important to keep in mind the difference between reactive and proactive data streams.

Reactive data streams are ones where the data comes to you. For instance, when a user visits your website, that's out of your control. You have no influence on the event. It just happens and you have to react to it.

Proactive data streams are ones where you have control on the data stream. For example, you might be reading the data from a file. You decide at which speed you want to read the data, in what order, etc.

If you consider data analysis as a whole, you're realize that the general approach is to turn reactive streams into proactive datasets. Events are usually logged into a database and are processed offline. Be it for building KPIs or training models.

The challenge for machine learning is to ensure models you train offline on proactive datasets will perform correctly in production on reactive data streams.

## Online processing

Online processing is the act of processing a data stream one element at a time. In the case of machine learning, that means training a model by teaching it one sample at a time. This is completely opposite to the traditional way of doing machine learning, which is to train a model on a whole batch data at a time.

An online model is therefore a stateful, dynamic object. It keeps learning and doesn't have to revisit past data. It's a different way of doing things, and therefore has its own set of pros and cons.

## Tasks

Machine learning encompasses many different tasks: classification, regression, anomaly detection, time series forecasting, etc. The ideology behind River is to be a generic machine learning which allows to perform these tasks in a streaming manner. Indeed, many batch machine learning algorithms have online equivalents.

Note that River also supports some more basic tasks. For instance, you might just want to calculate a running average of a data stream. These are usually smaller parts of a whole stream processing pipeline.

## Dictionaries everywhere

River is a Python library. It is composed of a bunch of classes which implement various online processing algorithms. Most of these classes are machine learning models which can process a single sample, be it for learning or for inference.

We made the choice to use dictionaries as the basic building block. First of all, online processing is different to batch processing, in that vectorization doesn't bring any speedup. Therefore numeric processing libraries such as numpy and PyTorch actually bring too much overhead. Using native Python data structures is faster.

Dictionaries are therefore a perfect fit. They're native to Python and have excellent support in the standard library. They allow naming each feature. They can hold any kind of data type. They allow transparent support of JSON payloads, allowing seemless integration with web apps.

## Datasets

In production, you're almost always going to face data streams which you have to react to. Such as users visiting your website. The advantage of online machine learning is that you can design models which make predictions as well as learn from this data stream as it flows.

But of course, when you're developping a model, you don't usually have access to a real-time feed on which to evaluate your model. You usually have an offline dataset which you want to evaluate your model on. River provides some datasets which can be read in online manner, one sample at a time. It is however crucial to keep in mind that the goal is to reproduce a production scenario as closely as possible, in order to ensure your model will perform just as well in production.

## Model evaluation

Online model evaluation differs from its traditional batch counterpart. In the latter, you usually perform cross-validation, whereby your training dataset is split into a learning and an evaluation dataset. This is fine, but it doesn't exactly reflect the data generation process that occurs in production.

Online model evaluation involves learning and inference in the same order as what would happen in production. Indeed, if you know the order in which your data arrives, then you can process it the exact same order. This allows you to replay a production scenario and evaluate your model with higher fidelity than cross-validation.

This is what makes online machine learning powerful. By replaying datasets in the correct order, you ensure you are designing models which will perform as expected in production.

## Concept drift

The main reason why an offline model might not perform as expected in production is because of concept drift. But this is true for all machine learning models, be they offline or online.

The advantage of online models over offline models is that they can cope with drift. Indeed, because they can keep learning, they usually adapt to concept drift in a seemless manner. As opposed to batch models which have to be retrained from scratch.
