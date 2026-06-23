---
description: How to install River, a Python library for online machine learning on streaming data.
---

# Installation

River is meant to work with Python 3.11 and above. Installation can be done via `pip`:

```sh
pip install river
```

And of course with uv:

```sh
uv add river
```

You can install the latest development version from GitHub, as so:

```sh
pip install git+https://github.com/online-ml/river --upgrade
pip install git+ssh://git@github.com/online-ml/river.git --upgrade  # using SSH
```

This method requires having Cython and Rust installed on your machine.

## Mini-batch support (optional `pandas` extra)

River's core online interface (`learn_one` / `predict_one`) does **not** require `pandas`. The mini-batch interface (`learn_many`, `predict_many`, `predict_proba_many`, `transform_many`) is built on top of `pandas.DataFrame` and `pandas.Series`, so `pandas` is an opt-in dependency.

To install River together with `pandas`:

```sh
pip install "river[pandas]"
# or
uv add "river[pandas]"
```

If you call a mini-batch method without `pandas` installed, River raises an `ImportError` pointing you to this extra.

Feel welcome to [open an issue on GitHub](https://github.com/online-ml/river/issues/new) if you are having any trouble.
