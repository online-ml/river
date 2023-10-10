from __future__ import annotations

from model_adapters.vw import VW2RiverClassifier
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
                    loss="log_loss", learning_rate="constant", eta0=LEARNING_RATE, penalty=None
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
        # Baseline
        "[baseline] Mean predictor": dummy.StatisticRegressor(stats.Mean()),
    },
}
