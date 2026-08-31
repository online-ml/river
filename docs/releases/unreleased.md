# Unreleased

## naive_bayes

- `MultinomialNB.learn_many`, `predict_many`, and `predict_proba_many` now accept any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only, preserving the input backend (including the pandas index) on output. A backend-agnostic `BaseNB.predict_many` was added so the argmax-over-probabilities logic is shared by all Naive Bayes variants.
