# Saving and loading models

River models are plain Python objects, so you can save and load them with the standard library's `pickle` module. This recipe covers serialization, deserialization, and the difference between `pickle`, `deepcopy`, and `clone`.

## Saving with pickle

Let's train a model for a bit and then save it to disk:

```python
import pickle
from river import datasets, linear_model, preprocessing

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

for x, y in datasets.Phishing():
    model.learn_one(x, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model saved ({__import__('os').path.getsize('model.pkl')} bytes)")
```

## Loading a saved model

Loading restores the full model state — all learned parameters, scaler statistics, etc.:

```python
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Verify it works
x, _ = next(iter(datasets.Phishing()))
loaded_model.predict_proba_one(x)
```

The loaded model is ready to predict and can continue learning from new data — no retraining needed.

## Serializing to bytes

If you don't want to write to a file (e.g. for storing in a database or sending over a network), use `pickle.dumps` and `pickle.loads`:

```python
data = pickle.dumps(model)
print(f"Serialized to {len(data)} bytes")

restored = pickle.loads(data)
restored.predict_proba_one(x)
```
