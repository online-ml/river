import pytest

from sklearn.utils import estimator_checks

from creme import cluster
from creme import compat
from creme import linear_model
from creme import naive_bayes
from creme import preprocessing


@pytest.mark.parametrize(
    'estimator',
    [
        pytest.param(estimator, id=str(estimator))
        for estimator in [
            naive_bayes.GaussianNB(),
            # multiclass.OneVsRestClassifier(
            #     binary_classifier=linear_model.LogisticRegression()
            # ),
            linear_model.LinearRegression(),
            preprocessing.StandardScaler(),
            preprocessing.OneHotEncoder(),
            cluster.KMeans(random_state=42),
            preprocessing.MinMaxScaler(),
            preprocessing.MinMaxScaler() + preprocessing.StandardScaler(),
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
        ]
    ]
)
def test_check_estimator(estimator):
    estimator_checks.check_estimator(compat.convert_creme_to_sklearn(estimator))
