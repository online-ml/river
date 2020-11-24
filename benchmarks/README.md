# Benchmarks

There are two kinds of benchmarks.

The first kind are notebooks, which are used to compare against other libraries.

The second kind are based on [ASV](https://github.com/airspeed-velocity/asv), and are meant to catch performance regressions.

## Notebooks

To run these benchmarks, navigate to this directory and create a `conda` virtual environment, as so:

```sh
$ conda create -n river-benchmarks -y python==3.8.5
$ conda activate river-benchmarks
$ pip install -r requirements.txt
$ conda install -c conda-forge -y vowpalwabbit
```

You may then run `jupyter lab` and open the notebooks.

Note that this will install the development version of `river` will be installed from GitHub. You may change this behaviour by modifying `requirements.txt` before creating the virtual environment.

Depending on what you want to do, you might have to run the `download_data.sh` script to obtain the necessary data.

## ASV

These benchmarks are located in the [`benchmarks`](benchmarks) subdirectory. You can run with the `asv` command-line tool. As a developer, you should run these tests on your laptop when you're working on a feature that is expected to affect performance, as so:

```sh
$ make bench
```

This will run the benchmarks for your latest local commit. It will also run the benchmarks against the `master` branch. It will then invoke the `asv compare` command to compare both versions of the code.

Check out the [ASV user guide](https://asv.readthedocs.io/en/stable/using.html) for more information.
