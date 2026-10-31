# iter\_arff

Iterates over rows from an ARFF file.

## Parameters

- **filepath\_or\_buffer**

  *Type* → `FilePath | typing.TextIO`

  Either a string indicating the location of a file, or a buffer object that has a `read` method.
- **target**

  *Type* → `str | list[str] | None`

  *Default* → `None`

  Name(s) of the target field. If `None`, then the target field is ignored. If a list of names is passed, then a dictionary is returned instead of a single value.
- **compression**

  *Type* → `Compression | None`

  *Default* → `infer`

  For on-the-fly decompression of on-disk data. If this is set to 'infer' and `filepath_or_buffer` is a path, then the decompression method is inferred for the following extensions: '.gz', '.zip'.
- **sparse**

  *Type* → `bool`

  *Default* → `False`

  Whether the data is sparse or not.

## Examples

```python
cars = '''
@relation CarData
@attribute make {Toyota, Honda, Ford, Chevrolet}
@attribute model string
@attribute year numeric
@attribute price numeric
@attribute mpg numeric
@data
Toyota, Corolla, 2018, 15000, 30.5
Honda, Civic, 2019, 16000, 32.2
Ford, Mustang, 2020, 25000, 25.0
Chevrolet, Malibu, 2017, 18000, 28.9
Toyota, Camry, 2019, 22000, 29.8
'''
with open('cars.arff', mode='w') as f:
    _ = f.write(cars)

from river import stream

for x, y in stream.iter_arff('cars.arff', target='price'):
    print(x, y)
```

```text
{'make': 'Toyota', 'model': ' Corolla', 'year': 2018.0, 'mpg': 30.5} 15000.0
{'make': 'Honda', 'model': ' Civic', 'year': 2019.0, 'mpg': 32.2} 16000.0
{'make': 'Ford', 'model': ' Mustang', 'year': 2020.0, 'mpg': 25.0} 25000.0
{'make': 'Chevrolet', 'model': ' Malibu', 'year': 2017.0, 'mpg': 28.9} 18000.0
{'make': 'Toyota', 'model': ' Camry', 'year': 2019.0, 'mpg': 29.8} 22000.0
```

Finally, let's delete the example file.

```python
import os; os.remove('cars.arff')
```

ARFF files support sparse data. Let's create a sparse ARFF file.

```python
sparse = '''
% traindata
@RELATION "traindata: -C 6"
@ATTRIBUTE y0 {0, 1}
@ATTRIBUTE y1 {0, 1}
@ATTRIBUTE y2 {0, 1}
@ATTRIBUTE y3 {0, 1}
@ATTRIBUTE y4 {0, 1}
@ATTRIBUTE y5 {0, 1}
@ATTRIBUTE X0 NUMERIC
@ATTRIBUTE X1 NUMERIC
@ATTRIBUTE X2 NUMERIC
@DATA
{ 3 1,6 0.863382,8 0.820094 }
{ 2 1,6 0.659761 }
{ 0 1,3 1,6 0.437881,8 0.818882 }
{ 2 1,6 0.676477,7 0.724635,8 0.755123 }
'''

with open('sparse.arff', mode='w') as f:
    _ = f.write(sparse)
```

In addition, we'll specify that there are several target fields.

```python
arff_stream = stream.iter_arff(
    'sparse.arff',
    target=['y0', 'y1', 'y2', 'y3', 'y4', 'y5'],
    sparse=True
)

for x, y in arff_stream:
    print(x)
    print(y)
```

```text
{'X0': '0.863382', 'X2': '0.820094'}
{'y0': 0, 'y1': 0, 'y2': 0, 'y3': '1', 'y4': 0, 'y5': 0}
{'X0': '0.659761'}
{'y0': 0, 'y1': 0, 'y2': '1', 'y3': 0, 'y4': 0, 'y5': 0}
{'X0': '0.437881', 'X2': '0.818882'}
{'y0': '1', 'y1': 0, 'y2': 0, 'y3': '1', 'y4': 0, 'y5': 0}
{'X0': '0.676477', 'X1': '0.724635', 'X2': '0.755123'}
{'y0': 0, 'y1': 0, 'y2': '1', 'y3': 0, 'y4': 0, 'y5': 0}
```

This function can also deal with missing features in non-sparse data. These are indicated with a question mark.

```python
data = '''
@relation giveMeLoan-weka.filters.unsupervised.attribute.Remove-R1
@attribute RevolvingUtilizationOfUnsecuredLines numeric
@attribute age numeric
@attribute NumberOfTime30-59DaysPastDueNotWorse numeric
@attribute DebtRatio numeric
@attribute MonthlyIncome numeric
@attribute NumberOfOpenCreditLinesAndLoans numeric
@attribute NumberOfTimes90DaysLate numeric
@attribute NumberRealEstateLoansOrLines numeric
@attribute NumberOfTime60-89DaysPastDueNotWorse numeric
@attribute NumberOfDependents numeric
@attribute isFraud {0,1}
@data
0.213179,74,0,0.375607,3500,3,0,1,0,1,0
0.305682,57,0,5710,?,8,0,3,0,0,0
0.754464,39,0,0.20994,3500,8,0,0,0,0,0
0.116951,27,0,46,?,2,0,0,0,0,0
0.189169,57,0,0.606291,23684,9,0,4,0,2,0
'''

with open('data.arff', mode='w') as f:
    _ = f.write(data)

for x, y in stream.iter_arff('data.arff', target='isFraud'):
    print(len(x))
```

```text
10
9
10
9
10
```

## References

- [ARFF format description from Weka](<https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/>)
