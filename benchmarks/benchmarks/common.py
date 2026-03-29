from __future__ import annotations

from river import (
    datasets,
    dummy,
    ensemble,
    evaluate,
    forest,
    linear_model,
    metrics,
    naive_bayes,
    neighbors,
    neural_net,
    optim,
    preprocessing,
    rules,
    stats,
    tree,
)

LEARNING_RATE = 0.005

# --- Benchmark class settings ---

BENCH_SETTINGS = dict(
    timeout=600,
    number=1,
    repeat=1,
    rounds=1,
    min_run_count=1,
    warmup_time=0,
)

# --- Binary classification ---

BINARY_CLF_DATASETS = [
    "Bananas",
    "Phishing",
]

_BINARY_CLF_MODEL_REGISTRY = {
    "Logistic regression": lambda: (
        preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
    ),
    "Aggregated Mondrian Forest": lambda: forest.AMFClassifier(seed=42),
    "ALMA": lambda: preprocessing.StandardScaler() | linear_model.ALMAClassifier(),
}

_BINARY_CLF_DATASET_REGISTRY = {
    "Bananas": lambda: datasets.Bananas(),
    "Phishing": lambda: datasets.Phishing(),
}

# --- Multiclass classification ---

MULTICLASS_CLF_DATASETS = [
    "ImageSegments",
]

_MULTICLASS_CLF_MODEL_REGISTRY = {
    "Naive Bayes": lambda: naive_bayes.GaussianNB(),
    "Hoeffding Tree": lambda: tree.HoeffdingTreeClassifier(),
    "Hoeffding Adaptive Tree": lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=42),
    "Adaptive Random Forest": lambda: forest.ARFClassifier(seed=42),
    "Aggregated Mondrian Forest": lambda: forest.AMFClassifier(seed=42),
    "Streaming Random Patches": lambda: ensemble.SRPClassifier(),
    "k-Nearest Neighbors": lambda: preprocessing.StandardScaler() | neighbors.KNNClassifier(),
    "ADWIN Bagging": lambda: ensemble.ADWINBaggingClassifier(
        tree.HoeffdingTreeClassifier(), seed=42
    ),
    "AdaBoost": lambda: ensemble.AdaBoostClassifier(tree.HoeffdingTreeClassifier(), seed=42),
    "Bagging": lambda: ensemble.BaggingClassifier(
        tree.HoeffdingAdaptiveTreeClassifier(bootstrap_sampling=False), seed=42
    ),
    "Leveraging Bagging": lambda: ensemble.LeveragingBaggingClassifier(
        tree.HoeffdingTreeClassifier(), seed=42
    ),
    "Voting": lambda: ensemble.VotingClassifier(
        [
            preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
            naive_bayes.GaussianNB(),
            tree.HoeffdingTreeClassifier(),
            preprocessing.StandardScaler() | neighbors.KNNClassifier(),
        ]
    ),
    "[baseline] Last Class": lambda: dummy.NoChangeClassifier(),
}

_MULTICLASS_CLF_DATASET_REGISTRY = {
    "ImageSegments": lambda: datasets.ImageSegments(),
}

# --- Regression ---

REGRESSION_DATASETS = [
    "ChickWeights",
    "TrumpApproval",
]

_REGRESSION_MODEL_REGISTRY = {
    "Linear Regression": lambda: (
        preprocessing.StandardScaler() | linear_model.LinearRegression()
    ),
    "Linear Regression with l1 regularization": lambda: (
        preprocessing.StandardScaler() | linear_model.LinearRegression(l1=1.0)
    ),
    "Linear Regression with l2 regularization": lambda: (
        preprocessing.StandardScaler() | linear_model.LinearRegression(l2=1.0)
    ),
    "Passive-Aggressive Regressor, mode 1": lambda: (
        preprocessing.StandardScaler() | linear_model.PARegressor(mode=1)
    ),
    "Passive-Aggressive Regressor, mode 2": lambda: (
        preprocessing.StandardScaler() | linear_model.PARegressor(mode=2)
    ),
    "k-Nearest Neighbors": lambda: (
        preprocessing.StandardScaler() | neighbors.KNNRegressor()
    ),
    "Hoeffding Tree": lambda: (
        preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()
    ),
    "Hoeffding Adaptive Tree": lambda: (
        preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeRegressor(seed=42)
    ),
    "Stochastic Gradient Tree": lambda: tree.SGTRegressor(),
    "Adaptive Random Forest": lambda: (
        preprocessing.StandardScaler() | forest.ARFRegressor(seed=42)
    ),
    "Aggregated Mondrian Forest": lambda: forest.AMFRegressor(seed=42),
    "Adaptive Model Rules": lambda: (
        preprocessing.StandardScaler() | rules.AMRules()
    ),
    "Streaming Random Patches": lambda: (
        preprocessing.StandardScaler() | ensemble.SRPRegressor(seed=42)
    ),
    "Bagging": lambda: (
        preprocessing.StandardScaler()
        | ensemble.BaggingRegressor(
            model=tree.HoeffdingAdaptiveTreeRegressor(bootstrap_sampling=False), seed=42
        )
    ),
    "Exponentially Weighted Average": lambda: (
        preprocessing.StandardScaler()
        | ensemble.EWARegressor(
            models=[
                linear_model.LinearRegression(),
                tree.HoeffdingAdaptiveTreeRegressor(),
                neighbors.KNNRegressor(),
                rules.AMRules(),
            ],
        )
    ),
    "River MLP": lambda: (
        preprocessing.StandardScaler()
        | neural_net.MLPRegressor(
            hidden_dims=(5,),
            activations=(
                neural_net.activations.ReLU,
                neural_net.activations.ReLU,
                neural_net.activations.Identity,
            ),
            optimizer=optim.SGD(1e-3),
            seed=42,
        )
    ),
    "[baseline] Mean predictor": lambda: dummy.StatisticRegressor(stats.Mean()),
}

_REGRESSION_DATASET_REGISTRY = {
    "ChickWeights": lambda: datasets.ChickWeights(),
    "TrumpApproval": lambda: datasets.TrumpApproval(),
}

# --- Lookup helpers ---

_REGISTRIES = {
    "binary_clf": (_BINARY_CLF_MODEL_REGISTRY, _BINARY_CLF_DATASET_REGISTRY),
    "multiclass_clf": (_MULTICLASS_CLF_MODEL_REGISTRY, _MULTICLASS_CLF_DATASET_REGISTRY),
    "regression": (_REGRESSION_MODEL_REGISTRY, _REGRESSION_DATASET_REGISTRY),
}


def get_model(track, name):
    """Return a fresh (cloned) model instance."""
    return _REGISTRIES[track][0][name]()


def get_dataset(track, name):
    """Return a dataset instance."""
    return _REGISTRIES[track][1][name]()


def _slugify(name):
    """Turn a human-readable name into a valid Python identifier."""
    return (
        name.lower()
        .replace("[baseline] ", "baseline_")
        .replace("-", "_")
        .replace(",", "")
        .replace(" ", "_")
    )


def _make_setup(track, model_name, dataset_list):
    def setup(self, dataset_name):
        self.model = get_model(track, model_name)
        dataset = get_dataset(track, dataset_name)
        next(iter(dataset))
        self.dataset = get_dataset(track, dataset_name)
    return setup


def _make_time_method(metric_fn):
    def time_progressive_val_score(self, dataset_name):
        for _ in evaluate.iter_progressive_val_score(
            dataset=self.dataset,
            model=self.model,
            metric=metric_fn(),
            measure_time=False,
            measure_memory=False,
        ):
            pass
    return time_progressive_val_score


def make_benchmark_classes(track, datasets, metric_fn):
    """Generate one ASV benchmark class per model, with datasets as params.

    Returns a dict of {class_name: class} to be injected into the caller's
    module globals.
    """
    model_registry = _REGISTRIES[track][0]
    classes = {}
    for model_name in model_registry:
        slug = _slugify(model_name)
        cls_name = f"Track_{slug}"
        cls = type(
            cls_name,
            (),
            {
                **BENCH_SETTINGS,
                "params": [datasets],
                "param_names": ["dataset"],
                "setup": _make_setup(track, model_name, datasets),
                "time_progressive_val_score": _make_time_method(metric_fn),
            },
        )
        classes[cls_name] = cls
    return classes
