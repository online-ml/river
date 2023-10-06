# Benchmarks

## Installation

The recommended way to run the benchmarks is to create a dedicated environment for river and its contenders.

An easy way to achieve that is through [Anaconda](https://docs.conda.io/projects/miniconda/en/latest/). Here is an example of creating an environment for the benchmarks:

```sh
conda create --name river-benchmark python=3.10
```

The next step is to clone river if you have not done that already:

```sh
git clone https://github.com/online-ml/river
cd river
```

From the river folder you can run the following command to install the needed dependencies:

```sh
pip install ".[benchmarks]"
```

## Usage

The `run.py` script executes the benchmarks and creates the necessary .csv files for rendering the plots.

```sh
cd benchmarks
python run.py
```

The `render.py` renders the plots from the .csv files and moves them to the `docs/benchmarks` folder.

```sh
python render.py
```

## Notes: VolpalWabbit

Installing Volpal Wabbit (VW) can be tricky sometimes. That is especially true when using apple silicon. If cannot make the pip install guidelines from VW work a workaround is the following. When using anaconda, you can install the recommended dependencies utilized for building VW with conda. You can get more info [here](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Building#conda) about such dependencies. After that, `pip install volpalwabbit` should work just fine.
