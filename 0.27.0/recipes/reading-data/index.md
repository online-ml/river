# Reading data

In River, the features of a sample are stored inside a dictionary, which in Python is called a `dict` and is a native data structure. In other words, we don't use any sophisticated data structure, such as a `numpy.ndarray` or a `pandas.DataFrame`.

The main advantage of using plain `dict`s is that it removes the overhead that comes with using the aforementioned data structures. This is important in a streaming context because we want to be able to process many individual samples in rapid succession. Another advantage is that `dict`s allow us to give names to our features. Finally, `dict`s are not typed, and can therefore store heterogeneous data.

Another advantage which we haven't mentioned is that `dict`s play nicely with Python's standard library. Indeed, Python contains many tools that allow manipulating `dict`s. For instance, the `csv.DictReader` can be used to read a CSV file and convert each row to a `dict`. In fact, the [`stream.iter_csv`](<https://riverml.xyz/0.27.0/api/stream/iter-csv>) method from River is just a wrapper on top of `csv.DictReader` that adds a few bells and whistles.

River provides some out-of-the-box datasets to get you started.

```python
from river import datasets

dataset = datasets.Bikes()
dataset
```

```
```

```text
Bike sharing station information from the city of Toulouse.

The goal is to predict the number of bikes in [1;36m5[0m different bike stations from the city of
Toulouse.

References
----------
[1m[[0m^[1;36m1[0m[1m][0m: [1m[[0mA short introduction and conclusion to the OpenBikes [1;36m2016[0m Challenge[1m][0m[1m([0m[4;94mhttps://maxhalford.github.io/blog/openbikes-challenge/[0m[4;94m)[0m

      Name  Bikes                                                         
      Task  Regression                                                    
   Samples  [1;36m182[0m,[1;36m470[0m                                                       
  Features  [1;36m8[0m                                                             
    Sparse  [3;91mFalse[0m                                                         
      Path  [35m/home/runner/river_data/Bikes/[0m[95mtoulouse_bikes.csv[0m              
       URL  [4;94mhttps://maxhalford.github.io/files/datasets/toulouse_bikes.zip[0m
      Size  [1;36m12.52[0m MiB                                                     
Downloaded  [3;91mFalse[0m
```

Note that when we say "loaded", we don't mean that the actual data is read from the disk. On the contrary, the dataset is a streaming data that can be iterated over one sample at a time. In Python lingo, it's a [generator](<https://realpython.com/introduction-to-python-generators/>).

Let's take a look at the first sample:

```python
x, y = next(iter(dataset))
x
```

```text
Downloading https://maxhalford.github.io/files/datasets/toulouse_bikes.zip (1.12 MiB)
Uncompressing into /home/runner/river_data/Bikes
```

```
```

```text
[1m{[0m
    [32m'moment'[0m: [1;35mdatetime.datetime[0m[1m([0m[1;36m2016[0m, [1;36m4[0m, [1;36m1[0m, [1;36m0[0m, [1;36m0[0m, [1;36m7[0m[1m)[0m,
    [32m'station'[0m: [32m'metro-canal-du-midi'[0m,
    [32m'clouds'[0m: [1;36m75[0m,
    [32m'description'[0m: [32m'light rain'[0m,
    [32m'humidity'[0m: [1;36m81[0m,
    [32m'pressure'[0m: [1;36m1017.0[0m,
    [32m'temperature'[0m: [1;36m6.54[0m,
    [32m'wind'[0m: [1;36m9.3[0m
[1m}[0m
```

Each dataset is iterable, which means we can also do:

```python
for x, y in dataset:
    break
x
```

```
```

```text
[1m{[0m
    [32m'moment'[0m: [1;35mdatetime.datetime[0m[1m([0m[1;36m2016[0m, [1;36m4[0m, [1;36m1[0m, [1;36m0[0m, [1;36m0[0m, [1;36m7[0m[1m)[0m,
    [32m'station'[0m: [32m'metro-canal-du-midi'[0m,
    [32m'clouds'[0m: [1;36m75[0m,
    [32m'description'[0m: [32m'light rain'[0m,
    [32m'humidity'[0m: [1;36m81[0m,
    [32m'pressure'[0m: [1;36m1017.0[0m,
    [32m'temperature'[0m: [1;36m6.54[0m,
    [32m'wind'[0m: [1;36m9.3[0m
[1m}[0m
```

As we can see, the values have different types.

Under the hood, calling `for x, y in dataset` simply iterates over a file and parses each value appropriately. We can do this ourselves by using [`stream.iter_csv`](<https://riverml.xyz/0.27.0/api/stream/iter-csv>):

```python
from river import stream

X_y = stream.iter_csv(dataset.path)
x, y = next(X_y)
x, y
```

```
```

```text
[1m([0m
    [1m{[0m
        [32m'moment'[0m: [32m'2016-04-01 00:00:07'[0m,
        [32m'bikes'[0m: [32m'1'[0m,
        [32m'station'[0m: [32m'metro-canal-du-midi'[0m,
        [32m'clouds'[0m: [32m'75'[0m,
        [32m'description'[0m: [32m'light rain'[0m,
        [32m'humidity'[0m: [32m'81'[0m,
        [32m'pressure'[0m: [32m'1017.0'[0m,
        [32m'temperature'[0m: [32m'6.54'[0m,
        [32m'wind'[0m: [32m'9.3'[0m
    [1m}[0m,
    [3;35mNone[0m
[1m)[0m
```

There are a couple things that are wrong. First of all, the numeric features have not been casted into numbers. Indeed, by default, [`stream.iter_csv`](<https://riverml.xyz/0.27.0/api/stream/iter-csv>) assumes that everything is a string. A related issue is that the `moment` field hasn't been parsed into a `datetime`. Finally, the target field, which is `bikes`, hasn't been separated from the rest of the features. We can remedy these issues by setting a few parameters:

```python
X_y = stream.iter_csv(
    dataset.path,
    converters={
        'bikes': int,
        'clouds': int,
        'humidity': int,
        'pressure': float,
        'temperature': float,
        'wind': float
    },
    parse_dates={'moment': '%Y-%m-%d %H:%M:%S'},
    target='bikes'
)
x, y = next(X_y)
x, y
```

```
```

```text
[1m([0m
    [1m{[0m
        [32m'moment'[0m: [1;35mdatetime.datetime[0m[1m([0m[1;36m2016[0m, [1;36m4[0m, [1;36m1[0m, [1;36m0[0m, [1;36m0[0m, [1;36m7[0m[1m)[0m,
        [32m'station'[0m: [32m'metro-canal-du-midi'[0m,
        [32m'clouds'[0m: [1;36m75[0m,
        [32m'description'[0m: [32m'light rain'[0m,
        [32m'humidity'[0m: [1;36m81[0m,
        [32m'pressure'[0m: [1;36m1017.0[0m,
        [32m'temperature'[0m: [1;36m6.54[0m,
        [32m'wind'[0m: [1;36m9.3[0m
    [1m}[0m,
    [1;36m1[0m
[1m)[0m
```

That's much better. We invite you to take a look at the `stream` module to see for yourself what other methods are available. Note that River is first and foremost a machine learning library, and therefore isn't as much concerned about reading data as it is about statistical algorithms. We do however believe that the fact that we use dictionary gives you, the user, a lot of freedom and flexibility.

The `stream` module provides helper functions to read data from different formats. For instance, you can use the [`stream.iter_sklearn_dataset`](<https://riverml.xyz/0.27.0/api/stream/iter-sklearn-dataset>) function to turn any scikit-learn dataset into a stream.

```python
from sklearn import datasets

dataset = datasets.load_diabetes()

for x, y in stream.iter_sklearn_dataset(dataset):
    break

x, y
```

```
```

```text
[1m([0m
    [1m{[0m
        [32m'age'[0m: [1;36m0.038075906433423026[0m,
        [32m'sex'[0m: [1;36m0.05068011873981862[0m,
        [32m'bmi'[0m: [1;36m0.061696206518683294[0m,
        [32m'bp'[0m: [1;36m0.0218723855140367[0m,
        [32m's1'[0m: [1;36m-0.04422349842444599[0m,
        [32m's2'[0m: [1;36m-0.03482076283769895[0m,
        [32m's3'[0m: [1;36m-0.04340084565202491[0m,
        [32m's4'[0m: [1;36m-0.002592261998183278[0m,
        [32m's5'[0m: [1;36m0.019907486170462722[0m,
        [32m's6'[0m: [1;36m-0.01764612515980379[0m
    [1m}[0m,
    [1;36m151.0[0m
[1m)[0m
```

To conclude, let us shortly mention the difference between *proactive learning* and *reactive learning* in the specific context of online machine learning. When we loop over a data with a `for` loop, we have the control over the data and the order in which it arrives. We are proactive in the sense that we, the user, are asking for the data to arrive.

In contrast, in a reactive situation, we don't have control on the data arrival. A typical example of such a situation is a web server, where web requests arrive in an arbitrary order. This is a situation where River shines. For instance, in a [Flask](<https://flask.palletsprojects.com/en/1.1.x/>) application, you could define a route to make predictions with a River model as so:

```python
import flask

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    payload = flask.request.json
    river_model = load_model()
    return river_model.predict_proba_one(payload)
```

Likewise, a model can be updated whenever a request arrives as so:

```python
@app.route('/', methods=['POST'])
def learn():
    payload = flask.request.json
    river_model = load_model()
    river_model.learn_one(payload['features'], payload['target'])
    return {}, 201
```

To summarize, River can be used in many different ways. The fact that it uses dictionaries to represent features provides a lot of flexibility and space for creativity.
