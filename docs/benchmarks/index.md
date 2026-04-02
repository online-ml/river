# Benchmarks

River benchmarks track the timing performance of models across git history using [airspeed velocity (asv)](https://asv.readthedocs.io/).

**[View benchmark results](https://online-ml.github.io/river/benchmarks/){ target="_blank" }**

The benchmarks cover three tracks:

- **Binary classification** — Logistic regression, AMF, ALMA on Bananas, Phishing
- **Multiclass classification** — Naive Bayes, Hoeffding Trees, Random Forests, ensembles, KNN on ImageSegments
- **Regression** — Linear models, trees, forests, ensembles, MLP on ChickWeights, TrumpApproval

Each benchmark runs full progressive validation (`evaluate.iter_progressive_val_score`) and measures wall-clock time. See the [benchmarks README](https://github.com/online-ml/river/tree/main/benchmarks) for instructions on running benchmarks locally.
