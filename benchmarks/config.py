from sklearn.linear_model import SGDClassifier

import torch
from benchmarks.model_zoo.torch import PyTorchBinaryClassifier, PyTorchLogReg
from benchmarks.model_zoo.vw import VW2RiverClassifier
from river import preprocessing, linear_model, tree, naive_bayes, ensemble, neighbors, rules, neural_net, dummy
from river import compat, optim, evaluate, stats


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
        "ALMA": preprocessing.StandardScaler() | linear_model.ALMAClassifier(),
        "Stochastic Gradient Tree": tree.SGTClassifier(),
        # "onelearn AMFClassifier": (
        #     compat.SKL2RiverClassifier(
        #         AMFClassifier(
        #             classes=[False, True],
        #         ),
        #         classes=[False, True],
        #     )
        # ),
        "sklearn SGDClassifier": (
            preprocessing.StandardScaler()
            | compat.SKL2RiverClassifier(
                SGDClassifier(
                    loss="log", learning_rate="constant", eta0=LEARNING_RATE, penalty="none"
                ),
                classes=[False, True],
            )
        ),
        "PyTorch logistic regression": (
            preprocessing.StandardScaler()
            | PyTorchBinaryClassifier(
                network_func=PyTorchLogReg,
                loss=torch.nn.BCELoss(),
                optimizer_func=lambda params: torch.optim.SGD(params, lr=LEARNING_RATE),
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
        "Extremely Fast Decision Tree": tree.ExtremelyFastDecisionTreeClassifier(),
        "Adaptive Random Forest": ensemble.AdaptiveRandomForestClassifier(seed=42),
        "Streaming Random Patches": ensemble.SRPClassifier(),
        "k-Nearest Neighbors": preprocessing.StandardScaler()
        | neighbors.KNNClassifier(window_size=100),
        "ADWIN Bagging": ensemble.ADWINBaggingClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "AdaBoost": ensemble.AdaBoostClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "Bagging": ensemble.BaggingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(bootstrap_sampling=False),
            seed=42
        ),
        "Leveraging Bagging": ensemble.LeveragingBaggingClassifier(
            tree.HoeffdingTreeClassifier(), seed=42
        ),
        "Stacking": ensemble.StackingClassifier(
            [
                preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
                naive_bayes.GaussianNB(),
                tree.HoeffdingTreeClassifier(),
                preprocessing.StandardScaler() | neighbors.KNNClassifier(window_size=100),
            ],
            meta_classifier=ensemble.AdaptiveRandomForestClassifier(seed=42),
        ),
        "Voting": ensemble.VotingClassifier(
            [
                preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
                naive_bayes.GaussianNB(),
                tree.HoeffdingTreeClassifier(),
                preprocessing.StandardScaler() | neighbors.KNNClassifier(window_size=100),
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
        "k-Nearest Neighbors": preprocessing.StandardScaler()
        | neighbors.KNNRegressor(window_size=100),
        "Hoeffding Tree": preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(),
        "Hoeffding Adaptive Tree": preprocessing.StandardScaler()
        | tree.HoeffdingAdaptiveTreeRegressor(seed=42),
        "Stochastic Gradient Tree": tree.SGTRegressor(),
        "Adaptive Random Forest": preprocessing.StandardScaler()
        | ensemble.AdaptiveRandomForestRegressor(seed=42),
        "Adaptive Model Rules": preprocessing.StandardScaler()
        | rules.AMRules(),
        "Streaming Random Patches": preprocessing.StandardScaler() | ensemble.SRPRegressor(seed=42),
        "Bagging": preprocessing.StandardScaler() | ensemble.BaggingRegressor(
            model=tree.HoeffdingAdaptiveTreeRegressor(bootstrap_sampling=False),
            seed=42
        ),
        "Exponentially Weighted Average": preprocessing.StandardScaler()
        | ensemble.EWARegressor(
            models=[
                linear_model.LinearRegression(),
                tree.HoeffdingAdaptiveTreeRegressor(),
                neighbors.KNNRegressor(window_size=100),
                rules.AMRules(),
            ],
        ),
        "Multi-layer Perceptron": preprocessing.StandardScaler()
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