import pickle
from river import imblearn
from river import linear_model
from river import preprocessing

def test_pickling():
    # See this discussion https://github.com/online-ml/river/discussions/544

    model = (
        preprocessing.StandardScaler() |
        imblearn.HardSamplingClassifier(
            classifier=linear_model.LogisticRegression(),
            p=0.1,
            size=40,
            seed=42,
        )
    )

    pickle.dumps(model)
