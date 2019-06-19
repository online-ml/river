.. image:: _static/creme.svg
    :width: 400px
    :height: 240px
    :align: center

.. image:: https://img.shields.io/github/stars/creme-ml/creme.svg?style=social
   :target: https://github.com/creme-ml/creme
   :align: center


Introduction
============

``creme`` is a library for in\ **creme**\ ntal learning. Incremental learning is a machine learning
regime where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. ``creme``'s API is heavily inspired from
that of scikit-learn, enough so that users who are familiar with scikit-learn should feel right at
home.

Most machine learning algorithms (be it supervised or unsupervised) assume a batch regime. However
some of these algorithms have online variants. For example stochastic gradient descent is the
online version of gradient descent whilst incremental k-means clustering is the online adaptation
of k-means clustering. In general online algorithms perform slightly worse than their batch
counterparts, although the gap is usually very small. However, online learning algorithms only
consume a tiny amount of RAM, which thus makes them scalable and ideal candidates for commodity
hardware and embedded systems. Moreover, they are much easier to put into production.

``creme`` provides a nice interface for putting an incremental learning pipeline in place; a bit
like what scikit-learn does for batch learning. Of course there are other open-source solutions
available, but they are somewhat specialized towards certain tasks and can require a steep learning
curve. Moreover some of these solutions aren't "truly online" as they mostly assume the data is
contained in a file. With ``creme`` it is possible to learn from a stream in it's largest sense, be
it a database query or a Kafka instance.


Contents
========

.. toctree::
   :maxdepth: 2

   install
   api
   user-guide
   faq
