# OutputCodeClassifier

Output-code multiclass strategy.

This also referred to as "error-correcting output codes".

This class allows to learn a multi-class classification problem with a binary classifier. Each class is converted to a code of 0s and 1s. The length of the code is called the code size. A copy of the classifier made for code. The codes associated with the classes are stored in a code book.

When a new sample arrives, the label's code is retrieved from the code book. Then, each classifier is trained on the relevant part of code, which is either a 0 or a 1.

For predicting, each classifier outputs a probability. These are then compared to each code in the code book, and the label which is the "closest" is chosen as the most likely class. Closeness is determined in terms of Manhattan distance.

One specificity of online learning is that we don't how many classes there are initially. Therefore, a random procedure generates random codes on the fly whenever a previously unseed label appears.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  A binary classifier, although a multi-class classifier will work too.
- **code\_size**

  *Type* → `int`

  The code size, which dictates how many copies of the provided classifiers to train. Must be strictly positive.
- **coding\_method**

  *Type* → `str`

  *Default* → `random`

  The method used to generate the codes. Can be either 'exact' or 'random'. The 'exact' method generates all possible codes of a given size in memory, and streams them in a random order. The 'random' method generates random codes of a given size on the fly. The 'exact' method necessarily generates different codes for each class, but requires more memory. The 'random' method can generate duplicate codes for different classes, but requires less memory.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  A random seed number that can be set for reproducibility.

## Examples

```python
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import multiclass
from river import preprocessing

dataset = datasets.ImageSegments()

scaler = preprocessing.StandardScaler()
ooc = multiclass.OutputCodeClassifier(
    classifier=linear_model.LogisticRegression(),
    code_size=10,
    coding_method='random',
    seed=1
)
model = scaler | ooc

metric = metrics.MacroF1()

evaluate.progressive_val_score(dataset, model, metric)
```

```text
MacroF1: 79.58%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## References

- [Dietterich, T.G. and Bakiri, G., 1994. Solving multiclass learning problems via error-correcting output codes. Journal of artificial intelligence research, 2, pp.263-286.](<https://arxiv.org/pdf/cs/9501101.pdf>)
- [James, G. and Hastie, T., 1998. The error coding method and PICTs. Journal of Computational and Graphical statistics, 7(3), pp.377-387.](<https://www.semanticscholar.org/paper/The-Error-Coding-Method-and-PICTs-James-Hastie/73abc83ed720921ed912a709aab6b734915b1012>)
