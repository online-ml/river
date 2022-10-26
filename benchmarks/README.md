# Benchmarks

Navigate to the root of this repo and create a `conda` virtual environment, as so:

```sh
conda create -f environment.yml
conda activate river-benchmarks
```

Also install whichever River version you want.

Then run the benchmarks:

```sh
python run.py
```

This creates a `results.json` file. To generate the page that gets displayed in the docs, do this:

```sh
python render.py
```

This `render.py` script gets run anyway when the docs are built. See the [Makefile](../Makefile).

Update the environment.yml file if additional dependencies are installed:

```sh
conda env export --no-builds | grep -v "^prefix: " > environment.yml
```
