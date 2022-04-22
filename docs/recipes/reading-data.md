# Reading data

In `river`, the features of a sample are stored inside a dictionary, which in Python is called a `dict` and is a native data structure. In other words, we don't use any sophisticated data structure, such as a `numpy.ndarray` or a `pandas.DataFrame`.

The main advantage of using plain `dict`s is that it removes the overhead that comes with using the aforementioned data structures. This is important in a streaming context because we want to be able to process many individual samples in rapid succession. Another advantage is that `dict`s allow us to give names to our features. Finally, `dict`s are not typed, and can therefore store heterogeneous data.

Another advantage which we haven't mentioned is that `dict`s play nicely with Python's standard library. Indeed, Python contains many tools that allow manipulating `dict`s. For instance, the `csv.DictReader` can be used to read a CSV file and convert each row to a `dict`. In fact, the `stream.iter_csv` method from `river` is just a wrapper on top of `csv.DictReader` that adds a few bells and whistles.

`river` provides some out-of-the-box datasets to get you started.


```python
from river import datasets

dataset = datasets.Bikes()
dataset
```




    Bike sharing station information from the city of Toulouse.
    
    The goal is to predict the number of bikes in 5 different bike stations from the city of
    Toulouse.
    
          Name  Bikes                                                         
          Task  Regression                                                    
       Samples  182,470                                                       
      Features  8                                                             
        Sparse  False                                                         
          Path  /Users/max.halford/river_data/Bikes/toulouse_bikes.csv        
           URL  https://maxhalford.github.io/files/datasets/toulouse_bikes.zip
          Size  12.52 MB                                                      
    Downloaded  True                                                          



Note that when we say "loaded", we don't mean that the actual data is read from the disk. On the contrary, the dataset is a streaming data that can be iterated over one sample at a time. In Python lingo, it's a [generator](https://realpython.com/introduction-to-python-generators/).

Let's take a look at the first sample:


```python
x, y = next(iter(dataset))
x
```




    {'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
     'station': 'metro-canal-du-midi',
     'clouds': 75,
     'description': 'light rain',
     'humidity': 81,
     'pressure': 1017.0,
     'temperature': 6.54,
     'wind': 9.3}



Each dataset is iterable, which means we can also do:


```python
for x, y in dataset:
    break
x
```




    {'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
     'station': 'metro-canal-du-midi',
     'clouds': 75,
     'description': 'light rain',
     'humidity': 81,
     'pressure': 1017.0,
     'temperature': 6.54,
     'wind': 9.3}



As we can see, the values have different types.

Under the hood, calling `for x, y in dataset` simply iterates over a file and parses each value appropriately. We can do this ourselves by using `stream.iter_csv`:


```python
from river import stream

X_y = stream.iter_csv(dataset.path)
x, y = next(X_y)
x, y
```




    ({'moment': '2016-04-01 00:00:07',
      'bikes': '1',
      'station': 'metro-canal-du-midi',
      'clouds': '75',
      'description': 'light rain',
      'humidity': '81',
      'pressure': '1017.0',
      'temperature': '6.54',
      'wind': '9.3'},
     None)



There are a couple things that are wrong. First of all, the numeric features have not been casted into numbers. Indeed, by default, `stream.iter_csv` assumes that everything is a string. A related issue is that the `moment` field hasn't been parsed into a `datetime`. Finally, the target field, which is `bikes`, hasn't been separated from the rest of the features. We can remedy to these issues by setting a few parameters:


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




    ({'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
      'station': 'metro-canal-du-midi',
      'clouds': 75,
      'description': 'light rain',
      'humidity': 81,
      'pressure': 1017.0,
      'temperature': 6.54,
      'wind': 9.3},
     1)



That's much better. We invite you to take a look at the `stream` module to see for yourself what other methods are available. Note that `river` is first and foremost a machine learning library, and therefore isn't as much concerned about reading data as it is about statistical algorithms. We do however believe that the fact that we use dictionary gives you, the user, a lot of freedom and flexibility.

The `stream` module provides helper functions to read data from different formats. For instance, you can use the `stream.iter_sklearn_dataset` function to turn any scikit-learn dataset into a stream.


```python
from sklearn import datasets

dataset = datasets.load_diabetes()

for x, y in stream.iter_sklearn_dataset(dataset):
    break

x, y
```




    ({'age': 0.0380759064334241,
      'sex': 0.0506801187398187,
      'bmi': 0.0616962065186885,
      'bp': 0.0218723549949558,
      's1': -0.0442234984244464,
      's2': -0.0348207628376986,
      's3': -0.0434008456520269,
      's4': -0.00259226199818282,
      's5': 0.0199084208763183,
      's6': -0.0176461251598052},
     151.0)



To conclude, let us shortly mention the difference between *proactive learning* and *reactive learning* in the specific context of online machine learning. When we loop over a data with a `for` loop, we have the control over the data and the order in which it arrives. We are proactive in the sense that we, the user, are asking for the data to arrive.

In contract, in a reactive situation, we don't have control on the data arrival. A typical example of such a situation is a web server, where web requests arrive in an arbitrary order. This is a situation where `river` shines. For instance, in a [Flask](https://flask.palletsprojects.com/en/1.1.x/) application, you could define a route to make predictions with a `river` model as so:


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

To summarize, `river` can be used in many different ways. The fact that it uses dictionaries to represent features provides a lot of flexibility and space for creativity.
