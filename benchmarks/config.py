from model_zoo.torch import (
    TorchLinearRegression,
    TorchLogisticRegression,
    TorchLSTMClassifier,
    TorchLSTMRegressor,
    TorchMLPClassifier,
    TorchMLPRegressor,
)
from model_zoo.vw import VW2RiverClassifier
from river_torch.classification import Classifier as TorchClassifier
from river_torch.classification import RollingClassifier as TorchRollingClassifier
from river_torch.regression import Regressor as TorchRegressor
from river_torch.regression import RollingRegressor as TorchRollingRegressor
from sklearn.linear_model import SGDClassifier

from river import (
    compat,
    dummy,
    ensemble,
    evaluate,
    forest,
    linear_model,
    naive_bayes,
    neighbors,
    neural_net,
    optim,
    preprocessing,
    rules,
    stats,
    tree,
)

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    evaluate.BinaryClassificationTrack(),
    evaluate.MultiClassClassificationTrack(),
    evaluate.RegressionTrack(),
]
import river

MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
        ),
        "Aggregated Mondrian Forest": forest.AMFClassifier(seed=42),
        "ALMA": preprocessing.StandardScaler() | linear_model.ALMAClassifier(),
        "sklearn SGDClassifier": (
            preprocessing.StandardScaler()
            | compat.SKL2RiverClassifier(
                SGDClassifier(
                    loss="log", learning_rate="constant", eta0=LEARNING_RATE, penalty="none"
                ),
                classes=[False, True],
            )
        ),
        "Vowpal Wabbit logistic regression": VW2RiverClassifier(
            sgd=True,
            learning_rate=LEARNING_RATE,
            loss_function="logistic",
            link="logistic",
            adaptive=False,
            normalized=False,
            invariant=False,
            l2=0.0,
            l1=0.0,
            power_t=0,
            quiet=True,
        ),
    },
    "Multiclass classification": {
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Hoeffding Adaptive Tree": tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "Adaptive Random Forest": forest.ARFClassifier(seed=42),
        "Aggregated Mondrian Forest": forest.AMFClassifier(seed=42),
        "Streaming Random Patches": ensemble.SRPClassifier(),
        "k-Nearest Neighbors": preprocessing.StandardScaler() | neighbors.KNNClassifier(),
        "ADWIN Bagging": ensemble.ADWINBaggingClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "AdaBoost": ensemble.AdaBoostClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "Bagging": ensemble.BaggingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(bootstrap_sampling=False), seed=42
        ),
        "Leveraging Bagging": ensemble.LeveragingBaggingClassifier(
            tree.HoeffdingTreeClassifier(), seed=42
        ),
        "Stacking": ensemble.StackingClassifier(
            [
                preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
                naive_bayes.GaussianNB(),
                tree.HoeffdingTreeClassifier(),
                preprocessing.StandardScaler() | neighbors.KNNClassifier(),
            ],
            meta_classifier=forest.ARFClassifier(seed=42),
        ),
        "Voting": ensemble.VotingClassifier(
            [
                preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
                naive_bayes.GaussianNB(),
                tree.HoeffdingTreeClassifier(),
                preprocessing.StandardScaler() | neighbors.KNNClassifier(),
            ]
        ),
        "Torch Logistic Regression": (
            preprocessing.StandardScaler()
            | TorchClassifier(
                module=TorchLogisticRegression,
                loss_fn="binary_cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Torch MLP": (
            preprocessing.StandardScaler()
            | TorchClassifier(
                module=TorchMLPClassifier,
                loss_fn="binary_cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Torch LSTM": (
            preprocessing.StandardScaler()
            | TorchRollingClassifier(
                module=TorchLSTMClassifier,
                loss_fn="binary_cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                lr=LEARNING_RATE,
                window_size=20,
                append_predict=False,
            )
        ),
        # Baseline
        "[baseline] Last Class": dummy.NoChangeClassifier(),
    },
    "Regression": {
        "Linear Regression": preprocessing.StandardScaler() | linear_model.LinearRegression(),
        "Linear Regression with l1 regularization": preprocessing.StandardScaler()
        | linear_model.LinearRegression(l1=1.0),
        "Linear Regression with l2 regularization": preprocessing.StandardScaler()
        | linear_model.LinearRegression(l2=1.0),
        "Passive-Aggressive Regressor, mode 1": preprocessing.StandardScaler()
        | linear_model.PARegressor(mode=1),
        "Passive-Aggressive Regressor, mode 2": preprocessing.StandardScaler()
        | linear_model.PARegressor(mode=2),
        "k-Nearest Neighbors": preprocessing.StandardScaler() | neighbors.KNNRegressor(),
        "Hoeffding Tree": preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(),
        "Hoeffding Adaptive Tree": preprocessing.StandardScaler()
        | tree.HoeffdingAdaptiveTreeRegressor(seed=42),
        "Stochastic Gradient Tree": tree.SGTRegressor(),
        "Adaptive Random Forest": preprocessing.StandardScaler() | forest.ARFRegressor(seed=42),
        "Aggregated Mondrian Forest": forest.AMFRegressor(seed=42),
        "Adaptive Model Rules": preprocessing.StandardScaler() | rules.AMRules(),
        "Streaming Random Patches": preprocessing.StandardScaler() | ensemble.SRPRegressor(seed=42),
        "Bagging": preprocessing.StandardScaler()
        | ensemble.BaggingRegressor(
            model=tree.HoeffdingAdaptiveTreeRegressor(bootstrap_sampling=False), seed=42
        ),
        "Exponentially Weighted Average": preprocessing.StandardScaler()
        | ensemble.EWARegressor(
            models=[
                linear_model.LinearRegression(),
                tree.HoeffdingAdaptiveTreeRegressor(),
                neighbors.KNNRegressor(),
                rules.AMRules(),
            ],
        ),
        "Torch Linear Regression": (
            preprocessing.StandardScaler()
            | TorchRegressor(
                module=TorchLinearRegression,
                loss_fn="mse",
                optimizer_fn="adam",
                learning_rate=LEARNING_RATE,
            )
        ),
        "Torch MLP": (
            preprocessing.StandardScaler()
            | TorchRegressor(
                module=TorchMLPRegressor,
                loss_fn="mse",
                optimizer_fn="adam",
                learning_rate=LEARNING_RATE,
            )
        ),
        "River MLP": preprocessing.StandardScaler()
        | neural_net.MLPRegressor(
            hidden_dims=(5,),
            activations=(
                neural_net.activations.ReLU,
                neural_net.activations.ReLU,
                neural_net.activations.Identity,
            ),
            optimizer=optim.SGD(1e-3),
            seed=42,
        ),
        "Torch LSTM": (
            preprocessing.StandardScaler()
            | TorchRollingRegressor(
                module=TorchLSTMRegressor,
                loss_fn="mse",
                optimizer_fn="adam",
                learning_rate=LEARNING_RATE,
                window_size=20,
                append_predict=False,
            )
        ),
        # Baseline
        "[baseline] Mean predictor": dummy.StatisticRegressor(stats.Mean()),
    },
}
