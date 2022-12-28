# Benchmarks

## Installation 
```sh
pip install ".[benchmarks]"
```

## Usage
The `run.py` executes the benchmarks and creates the necessary .csv files for rendering the plots.
```sh
cd benchmarks
python run.py
```
The `render.py` renders the plots from the .csv files and moves them to the `docs/benchmarks` folder.
```sh
python render.py
```
