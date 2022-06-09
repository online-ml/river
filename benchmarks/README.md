# Benchmarks

Navigate to the root of this repo and create a `conda` virtual environment, as so:

```sh
conda create -n river-benchmarks -y python==3.8.5
conda activate river-benchmarks
pip install -e ".[benchmarks]"
```

Then run the benchmarks:

```sh
python run.py
```

This creates a `results.json` file. To generate the page that gets displayed in the docs, do this:

```sh
python render.py
```

This `render.py` script gets run anyway when the docs are built. See the [Makefile](../Makefile).
