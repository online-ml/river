# Cloning and mutating

Sometimes you might want to reset a model, or edit (what we call mutate) its attributes. This can be useful in an online environment. Indeed, if you detect a drift, then you might want to mutate a model's attributes. Or if you see that a model's performance is plummeting, then you might to reset it to its "factory settings".

Anyway, this is not to convince you, but rather to say that a model's attributes don't have be to set in stone throughout its lifetime. In particular, if you're developing your own model, then you might want to have good tools to do this. This is what this recipe is about.

## Cloning

The first thing you can do is clone a model. This creates a deep copy of the model. The resulting model is entirely independent of the original model. The clone is fresh, in the sense that it is as if it hasn't seen any data.

For instance, say you have a linear regression model which you have trained on some data.

```python
from river import datasets, linear_model, optim, preprocessing

model = (
    preprocessing.StandardScaler() |
    linear_model.LinearRegression(
        optimizer=optim.SGD(3e-2)
    )
)

for x, y in datasets.TrumpApproval():
    model.predict_one(x)
    model.learn_one(x, y)

model[-1].weights
```

```text
{'ordinal_date': 20.59955380229643,
 'gallup': 0.39114944304212645,
 'ipsos': 0.4101918314868111,
 'morning_consult': 0.12042970179504908,
 'rasmussen': 0.18951231512561392,
 'you_gov': 0.04991712783831687}
```

For whatever reason, we may want to clone this model. This can be done with the `clone` method.

```python
clone = model.clone()
clone[-1].weights
```

```text
{}
```

As we can see, there are no weights because the clone is fresh copy that has not seen any data. However, the learning rate we specified is preserved.

```python
clone[-1].optimizer.learning_rate
```

```text
0.03
```

You may also specify parameters you want changed. For instance, let's say we want to clone the model, but we want to change the optimizer:

```python
clone = model.clone({"LinearRegression": {"optimizer": optim.Adam()}})
clone[-1].optimizer
```

```text
Adam({'lr': Constant({'learning_rate': 0.1}), 'n_iterations': 0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-08, 'm': None, 'v': None})
```

The first key indicates that we want to specify a different parameter for the `LinearRegression` part of the pipeline. Then the second key accesses the linear regression's `optimizer` parameter.

Finally, note that the `clone` method isn't reserved to models. Indeed, every object in River has it. That's because they all inherit from the `Base` class in the `base` module.

## Mutating attributes

Cloning a model can be useful, but the fact that it essentially resets the model may not be desired. Instead, you might want to change a attribute while preserving the model's state. For example, let's change the `l2` attribute, and the optimizer's `lr` attribute.

```python
model.mutate({
    "LinearRegression": {
        "l2": 0.1,
        "optimizer": {
            "lr": optim.schedulers.Constant(25e-3)
        }
    }
})

print(repr(model))
```

```text
Pipeline (
  StandardScaler (
    with_std=True
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.025
      )
    )
    loss=Squared ()
    l2=0.1
    l1=0.
    intercept_init=0.
    intercept_lr=Constant (
      learning_rate=0.01
    )
    clip_gradient=1e+12
    initializer=Zeros ()
  )
)
```

We can see the attributes we specified have changed. However, the model's state is preserved:

```python
model[-1].weights
```

```text
{'ordinal_date': 20.59955380229643,
 'gallup': 0.39114944304212645,
 'ipsos': 0.4101918314868111,
 'morning_consult': 0.12042970179504908,
 'rasmussen': 0.18951231512561392,
 'you_gov': 0.04991712783831687}
```

In other words, the `mutate` method does not create a deep copy of the model. It just sets attributes. At this point you may ask:

> *Why can't I just change the attribute directly, without calling `mutate`?*

The answer is that you're free to do proceed as such, but it's not the way we recommend. The `mutate` method is safer, in that it prevents you from mutating attributes you shouldn't be touching. We call these *immutable attributes*. For instance, there's no reason you should be modifying the weights.

```python
try:
    model.mutate({
        "LinearRegression": {
            "weights": "this makes no sense"
        }
    })
except ValueError as e:
    print(e)
```

```text
'weights' is not a mutable attribute of LinearRegression
```

All attributes are immutable by default. Under the hood, each model can specify a set of mutable attributes via the `_mutable_attributes` property. In theory this can be overridden. But the general idea is that we will progressively add more and more mutable attributes with time.

And that concludes this recipe. Arguably, this recipe caters to advanced users, and in particular users who are developing their own models. And yet, one could also argue that modifying parameters of a model on-the-fly is a great tool to have at your disposal when you're doing online machine learning.
