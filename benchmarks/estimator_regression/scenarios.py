"""Python inventory for the estimator regression suite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn import linear_model as sklearn_linear_model

from benchmarks.estimator_regression import workloads
from river import (
    active,
    anomaly,
    bandit,
    cluster,
    compat,
    compose,
    conf,
    drift,
    dummy,
    ensemble,
    facto,
    forest,
    imblearn,
    linear_model,
    metrics,
    model_selection,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    optim,
    preprocessing,
    reco,
    rules,
    stats,
    time_series,
    tree,
)


@dataclass(frozen=True)
class Scenario:
    """A deterministic estimator run used for base/head regression checks.

    Attributes:
        id: Permanent scenario identifier. This must be exactly
            ``"<module>.<ClassName>"`` for the public River estimator being
            covered. The audit derives ``"river.<id>"`` from this value.
        harness: Name of the run protocol used by ``run.py``.
        workload: Deterministic workload ID from ``workloads.py``.
        build: Factory that returns a fresh estimator instance for each run.
        n_samples: Number of samples materialized from the workload.
    """

    id: str
    harness: str
    workload: str
    build: Callable[[], Any]
    n_samples: int = workloads.N_SAMPLES

    @property
    def estimator(self) -> str:
        """Fully qualified public River estimator covered by this scenario."""

        return f"river.{self.id}"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="active.EntropySampler",
        harness="behavioral_invariant",
        workload="binary_sea_v1",
        build=lambda: active.EntropySampler(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)), seed=42
        ),
    ),
    Scenario(
        id="anomaly.HalfSpaceTrees",
        harness="anomaly",
        workload="anomaly_mixture_v1",
        build=lambda: anomaly.HalfSpaceTrees(n_trees=25, height=15, window_size=250, seed=42),
    ),
    Scenario(
        id="anomaly.LODA",
        harness="anomaly",
        workload="anomaly_mixture_v1",
        build=lambda: anomaly.LODA(n_bins=10, n_random_cuts=5, seed=42),
    ),
    Scenario(
        id="anomaly.LocalOutlierFactor",
        harness="anomaly",
        workload="anomaly_mixture_v1",
        build=lambda: anomaly.LocalOutlierFactor(),
    ),
    Scenario(
        id="anomaly.OneClassSVM",
        harness="anomaly",
        workload="anomaly_mixture_v1",
        build=lambda: anomaly.OneClassSVM(),
    ),
    Scenario(
        id="cluster.CluStream",
        harness="clustering",
        workload="cluster_blobs_v1",
        build=lambda: cluster.CluStream(
            n_macro_clusters=3, max_micro_clusters=50, time_window=100, time_gap=2, seed=42
        ),
    ),
    Scenario(
        id="cluster.DBSTREAM",
        harness="clustering",
        workload="cluster_blobs_v1",
        build=lambda: cluster.DBSTREAM(),
    ),
    Scenario(
        id="cluster.DenStream",
        harness="clustering",
        workload="cluster_blobs_v1",
        build=lambda: cluster.DenStream(),
    ),
    Scenario(
        id="cluster.KMeans",
        harness="clustering",
        workload="cluster_blobs_v1",
        build=lambda: cluster.KMeans(n_clusters=3, seed=42),
    ),
    Scenario(
        id="cluster.STREAMKMeans",
        harness="clustering",
        workload="cluster_blobs_v1",
        build=lambda: cluster.STREAMKMeans(),
    ),
    Scenario(
        id="compat.SKL2RiverClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compat.SKL2RiverClassifier(
            estimator=sklearn_linear_model.SGDClassifier(loss="log_loss"), classes=[False, True]
        ),
    ),
    Scenario(
        id="compat.SKL2RiverRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: compat.SKL2RiverRegressor(
            estimator=sklearn_linear_model.SGDRegressor(tol=1e-10)
        ),
    ),
    Scenario(
        id="conf.RegressionJackknife",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: conf.RegressionJackknife(regressor=linear_model.LinearRegression()),
    ),
    Scenario(
        id="drift.DriftRetrainingClassifier",
        harness="behavioral_invariant",
        workload="binary_sea_v1",
        build=lambda: drift.DriftRetrainingClassifier(
            model=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            drift_detector=drift.binary.DDM(),
            train_in_background=False,
        ),
    ),
    Scenario(
        id="dummy.NoChangeClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: dummy.NoChangeClassifier(),
    ),
    Scenario(
        id="dummy.PriorClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: dummy.PriorClassifier(),
    ),
    Scenario(
        id="dummy.StatisticRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: dummy.StatisticRegressor(statistic=stats.Mean()),
    ),
    Scenario(
        id="ensemble.ADWINBaggingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), seed=42
        ),
    ),
    Scenario(
        id="ensemble.ADWINBoostingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.ADWINBoostingClassifier(
            model=tree.HoeffdingTreeClassifier(), seed=42
        ),
    ),
    Scenario(
        id="ensemble.AdaBoostClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.AdaBoostClassifier(model=tree.HoeffdingTreeClassifier(), seed=42),
    ),
    Scenario(
        id="ensemble.BOLEClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.BOLEClassifier(model=tree.HoeffdingTreeClassifier(), seed=42),
    ),
    Scenario(
        id="ensemble.BaggingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.BaggingClassifier(model=tree.HoeffdingTreeClassifier(), seed=42),
    ),
    Scenario(
        id="ensemble.BaggingRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.BaggingRegressor(model=tree.HoeffdingTreeRegressor(), seed=42),
    ),
    Scenario(
        id="ensemble.EWARegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.EWARegressor(
            models=[linear_model.LinearRegression(), linear_model.LinearRegression()]
        ),
    ),
    Scenario(
        id="ensemble.LeveragingBaggingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), seed=42
        ),
    ),
    Scenario(
        id="ensemble.SRPClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(), seed=42),
    ),
    Scenario(
        id="ensemble.SRPRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.SRPRegressor(model=tree.HoeffdingTreeRegressor(), seed=42),
    ),
    Scenario(
        id="ensemble.StackingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.StackingClassifier(
            models=[
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            ],
            meta_classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
        ),
    ),
    Scenario(
        id="ensemble.VotingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: ensemble.VotingClassifier(
            models=[
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            ]
        ),
    ),
    Scenario(
        id="facto.FMClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: facto.FMClassifier(seed=42),
    ),
    Scenario(
        id="facto.FMRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: facto.FMRegressor(seed=42),
    ),
    Scenario(
        id="facto.HOFMClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: facto.HOFMClassifier(seed=42),
    ),
    Scenario(
        id="facto.HOFMRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: facto.HOFMRegressor(seed=42),
    ),
    Scenario(
        id="forest.AMFClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: forest.AMFClassifier(seed=42),
    ),
    Scenario(
        id="forest.AMFRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: forest.AMFRegressor(seed=42),
    ),
    Scenario(
        id="forest.ARFClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: forest.ARFClassifier(seed=42),
    ),
    Scenario(
        id="forest.ARFRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: forest.ARFRegressor(seed=42),
    ),
    Scenario(
        id="forest.OXTRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: forest.OXTRegressor(seed=42),
    ),
    Scenario(
        id="imblearn.ChebyshevOverSampler",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: imblearn.ChebyshevOverSampler(regressor=linear_model.LinearRegression()),
    ),
    Scenario(
        id="imblearn.ChebyshevUnderSampler",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: imblearn.ChebyshevUnderSampler(
            regressor=linear_model.LinearRegression(), seed=42
        ),
    ),
    Scenario(
        id="imblearn.HardSamplingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: imblearn.HardSamplingClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            size=50,
            p=0.5,
            seed=42,
        ),
    ),
    Scenario(
        id="imblearn.HardSamplingRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: imblearn.HardSamplingRegressor(
            regressor=linear_model.LinearRegression(), size=50, p=0.5, seed=42
        ),
    ),
    Scenario(
        id="imblearn.RandomOverSampler",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: imblearn.RandomOverSampler(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            desired_dist={False: 0.5, True: 0.5},
            seed=42,
        ),
    ),
    Scenario(
        id="imblearn.RandomSampler",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: imblearn.RandomSampler(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            desired_dist={False: 0.5, True: 0.5},
            seed=42,
        ),
    ),
    Scenario(
        id="imblearn.RandomUnderSampler",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: imblearn.RandomUnderSampler(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            desired_dist={False: 0.5, True: 0.5},
            seed=42,
        ),
    ),
    Scenario(
        id="linear_model.ALMAClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.ALMAClassifier()
        ),
    ),
    Scenario(
        id="linear_model.AdPredictor",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compose.Pipeline(preprocessing.StandardScaler(), linear_model.AdPredictor()),
    ),
    Scenario(
        id="linear_model.BayesianLinearRegression",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: linear_model.BayesianLinearRegression(),
    ),
    Scenario(
        id="linear_model.LinearRegression",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: linear_model.LinearRegression(),
    ),
    Scenario(
        id="linear_model.LogisticRegression",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
        ),
    ),
    Scenario(
        id="linear_model.PAClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compose.Pipeline(preprocessing.StandardScaler(), linear_model.PAClassifier()),
    ),
    Scenario(
        id="linear_model.PARegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: linear_model.PARegressor(),
    ),
    Scenario(
        id="linear_model.Perceptron",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: compose.Pipeline(preprocessing.StandardScaler(), linear_model.Perceptron()),
    ),
    Scenario(
        id="linear_model.SoftmaxRegression",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: linear_model.SoftmaxRegression(),
    ),
    Scenario(
        id="model_selection.BanditClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: model_selection.BanditClassifier(
            models=[
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            ],
            metric=metrics.Accuracy(),
            policy=bandit.EpsilonGreedy(epsilon=0.1, seed=42),
        ),
    ),
    Scenario(
        id="model_selection.BanditRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: model_selection.BanditRegressor(
            models=[linear_model.LinearRegression(), linear_model.LinearRegression()],
            metric=metrics.MAE(),
            policy=bandit.EpsilonGreedy(epsilon=0.1, seed=42),
        ),
    ),
    Scenario(
        id="model_selection.GreedyRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: model_selection.GreedyRegressor(
            models=[linear_model.LinearRegression(), linear_model.LinearRegression()]
        ),
    ),
    Scenario(
        id="model_selection.SuccessiveHalvingClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        build=lambda: model_selection.SuccessiveHalvingClassifier(
            models=[
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
                linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            ],
            metric=metrics.Accuracy(),
            budget=100,
        ),
    ),
    Scenario(
        id="model_selection.SuccessiveHalvingRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: model_selection.SuccessiveHalvingRegressor(
            models=[linear_model.LinearRegression(), linear_model.LinearRegression()],
            metric=metrics.MAE(),
            budget=100,
        ),
    ),
    Scenario(
        id="multiclass.OneVsOneClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: multiclass.OneVsOneClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005))
        ),
    ),
    Scenario(
        id="multiclass.OneVsRestClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: multiclass.OneVsRestClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005))
        ),
    ),
    Scenario(
        id="multiclass.OutputCodeClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: multiclass.OutputCodeClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)),
            code_size=2,
            seed=42,
        ),
    ),
    Scenario(
        id="multioutput.ClassifierChain",
        harness="multilabel_classification",
        workload="multioutput_binary_v1",
        build=lambda: multioutput.ClassifierChain(
            model=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005))
        ),
    ),
    Scenario(
        id="multioutput.MonteCarloClassifierChain",
        harness="multilabel_classification",
        workload="multioutput_binary_v1",
        build=lambda: multioutput.MonteCarloClassifierChain(
            model=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005)), seed=42
        ),
    ),
    Scenario(
        id="multioutput.PerOutputClassifier",
        harness="multilabel_classification",
        workload="multioutput_binary_v1",
        build=lambda: multioutput.PerOutputClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005))
        ),
    ),
    Scenario(
        id="multioutput.PerOutputRegressor",
        harness="multitarget_regression",
        workload="multitarget_regression_v1",
        build=lambda: multioutput.PerOutputRegressor(model=linear_model.LinearRegression()),
    ),
    Scenario(
        id="multioutput.ProbabilisticClassifierChain",
        harness="multilabel_classification",
        workload="multioutput_binary_v1",
        build=lambda: multioutput.ProbabilisticClassifierChain(
            model=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.005))
        ),
    ),
    Scenario(
        id="multioutput.RegressorChain",
        harness="multitarget_regression",
        workload="multitarget_regression_v1",
        build=lambda: multioutput.RegressorChain(model=linear_model.LinearRegression()),
    ),
    Scenario(
        id="naive_bayes.BernoulliNB",
        harness="binary_classification",
        workload="categorical_5x20_v1",
        build=lambda: naive_bayes.BernoulliNB(),
    ),
    Scenario(
        id="naive_bayes.ComplementNB",
        harness="binary_classification",
        workload="categorical_5x20_v1",
        build=lambda: naive_bayes.ComplementNB(),
    ),
    Scenario(
        id="naive_bayes.GaussianNB",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: naive_bayes.GaussianNB(),
    ),
    Scenario(
        id="naive_bayes.MultinomialNB",
        harness="binary_classification",
        workload="categorical_5x20_v1",
        build=lambda: naive_bayes.MultinomialNB(),
    ),
    Scenario(
        id="neighbors.KNNClassifier",
        harness="binary_classification",
        workload="binary_sea_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: neighbors.KNNClassifier(),
    ),
    Scenario(
        id="neighbors.KNNRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: neighbors.KNNRegressor(),
    ),
    Scenario(
        id="reco.Baseline",
        harness="recommendation",
        workload="user_item_ratings_v1",
        build=lambda: reco.Baseline(seed=42),
    ),
    Scenario(
        id="reco.BiasedMF",
        harness="recommendation",
        workload="user_item_ratings_v1",
        build=lambda: reco.BiasedMF(seed=42),
    ),
    Scenario(
        id="reco.FunkMF",
        harness="recommendation",
        workload="user_item_ratings_v1",
        build=lambda: reco.FunkMF(seed=42, n_factors=5, optimizer=optim.SGD(lr=0.005), l2=0.1),
    ),
    Scenario(
        id="reco.RandomNormal",
        harness="recommendation",
        workload="user_item_ratings_v1",
        build=lambda: reco.RandomNormal(seed=42),
    ),
    Scenario(
        id="rules.AMRules",
        harness="regression",
        workload="regression_friedman_v1",
        n_samples=workloads.N_HEAVY_SAMPLES,
        build=lambda: rules.AMRules(),
    ),
    Scenario(
        id="time_series.HoltWinters",
        harness="forecasting",
        workload="seasonal_series_v1",
        build=lambda: time_series.HoltWinters(alpha=0.3, beta=0.1, gamma=0.1, seasonality=12),
    ),
    Scenario(
        id="time_series.SNARIMAX",
        harness="forecasting",
        workload="seasonal_series_v1",
        build=lambda: time_series.SNARIMAX(p=1, d=1, q=1, m=12),
    ),
    Scenario(
        id="tree.ExtremelyFastDecisionTreeClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: tree.ExtremelyFastDecisionTreeClassifier(),
    ),
    Scenario(
        id="tree.HoeffdingAdaptiveTreeClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=42),
    ),
    Scenario(
        id="tree.HoeffdingAdaptiveTreeRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: tree.HoeffdingAdaptiveTreeRegressor(seed=42),
    ),
    Scenario(
        id="tree.HoeffdingTreeClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: tree.HoeffdingTreeClassifier(),
    ),
    Scenario(
        id="tree.HoeffdingTreeRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: tree.HoeffdingTreeRegressor(),
    ),
    Scenario(
        id="tree.ISOUPTreeRegressor",
        harness="multitarget_regression",
        workload="multitarget_regression_v1",
        build=lambda: tree.ISOUPTreeRegressor(),
    ),
    Scenario(
        id="tree.LASTClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: tree.LASTClassifier(),
    ),
    Scenario(
        id="tree.SGTClassifier",
        harness="multiclass_classification",
        workload="multiclass_led_v1",
        build=lambda: tree.SGTClassifier(),
    ),
    Scenario(
        id="tree.SGTRegressor",
        harness="regression",
        workload="regression_friedman_v1",
        build=lambda: tree.SGTRegressor(),
    ),
)

EXCLUSIONS: dict[str, str] = {
    "river.anomaly.GaussianScorer": "Supervised scorer (score_one(x, y)); needs the label at scoring "
    "time, outside the unsupervised anomaly harness.",
    "river.anomaly.PredictiveAnomalyDetection": "Supervised anomaly detector wrapping a predictor; "
    "needs label-aware scoring.",
    "river.anomaly.QuantileFilter": "Anomaly filter that classifies a wrapped scorer's output; needs "
    "a scorer + label-aware protocol.",
    "river.anomaly.StandardAbsoluteDeviation": "Supervised scorer (score_one(x, y)); needs the label "
    "at scoring time.",
    "river.anomaly.ThresholdFilter": "Anomaly filter that classifies a wrapped scorer's output; needs "
    "a scorer + label-aware protocol.",
    "river.cluster.ODAC": "Hierarchical clustering builder; predict_one is unimplemented (no cluster "
    "assignment).",
    "river.cluster.TextClust": "Text micro-clusterer; needs a tokenised text workload.",
    "river.compose.Discard": "Composition helper, not an ML algorithm.",
    "river.compose.FuncTransformer": "Stateless function wrapper, not an ML algorithm.",
    "river.compose.Grouper": "Grouping helper, not an ML algorithm.",
    "river.compose.Pipeline": "Composition container, not a standalone ML algorithm.",
    "river.compose.Prefixer": "String-prefixing transformer, not an ML algorithm.",
    "river.compose.Renamer": "Feature-renaming transformer, not an ML algorithm.",
    "river.compose.Select": "Feature-selection-by-name helper, not an ML algorithm.",
    "river.compose.SelectType": "Type-filtering helper, not an ML algorithm.",
    "river.compose.Suffixer": "String-suffixing transformer, not an ML algorithm.",
    "river.compose.TargetTransformRegressor": "Target-transform meta-wrapper; needs user-supplied "
    "func/inverse_func, not a standalone algorithm.",
    "river.compose.TransformerProduct": "Composition container, not a standalone ML algorithm.",
    "river.compose.TransformerUnion": "Composition container, not a standalone ML algorithm.",
    "river.facto.FFMClassifier": "Field-aware factorisation machine; expects field=feature structured "
    "input, needs a dedicated field-aware workload.",
    "river.facto.FFMRegressor": "Field-aware factorisation machine; expects field=feature structured "
    "input, needs a dedicated field-aware workload.",
    "river.facto.FwFMClassifier": "Field-weighted factorisation machine; expects field=feature "
    "structured input, needs a dedicated field-aware workload.",
    "river.facto.FwFMRegressor": "Field-weighted factorisation machine; expects field=feature "
    "structured input, needs a dedicated field-aware workload.",
    "river.feature_extraction.Agg": "Streaming aggregate transformer, not an ML algorithm.",
    "river.feature_extraction.BagOfWords": "Text vectoriser, not an ML algorithm.",
    "river.feature_extraction.PolynomialExtender": "Feature expander, not an ML algorithm.",
    "river.feature_extraction.RBFSampler": "Kernel approximation transformer, not an ML algorithm.",
    "river.feature_extraction.RandomTreesEmbedding": "Tree-based embedding transformer, not an ML "
    "algorithm.",
    "river.feature_extraction.TFIDF": "Text vectoriser transformer, not an ML algorithm.",
    "river.feature_extraction.TargetAgg": "Streaming target-aggregate transformer, not an ML "
    "algorithm.",
    "river.feature_selection.PoissonInclusion": "Randomised feature selector, not an ML algorithm.",
    "river.feature_selection.SelectKBest": "Feature selector, not an ML algorithm.",
    "river.feature_selection.VarianceThreshold": "Feature selector, not an ML algorithm.",
    "river.misc.ZstdClassifier": "Requires Python 3.14 compression.zstd; CI runs Python 3.13.",
    "river.multioutput.MultiClassEncoder": "Raises KeyError on predict-before-learn (encoder label "
    "map is empty until the first learn); incompatible with "
    "the online predict-then-learn protocol.",
    "river.neighbors.LazySearch": "Nearest-neighbour search index, not a predictor.",
    "river.neighbors.SWINN": "Nearest-neighbour search graph, not a predictor.",
    "river.preprocessing.AdaptiveStandardScaler": "Feature scaler, not an ML algorithm.",
    "river.preprocessing.Binarizer": "Feature binariser, not an ML algorithm.",
    "river.preprocessing.FeatureHasher": "Feature hasher, not an ML algorithm.",
    "river.preprocessing.GaussianRandomProjector": "Random projection transformer, not an ML "
    "algorithm.",
    "river.preprocessing.LDA": "Linear discriminant projection transformer, not an ML algorithm.",
    "river.preprocessing.MaxAbsScaler": "Feature scaler, not an ML algorithm.",
    "river.preprocessing.MinMaxScaler": "Feature scaler, not an ML algorithm.",
    "river.preprocessing.Normalizer": "Feature normaliser, not an ML algorithm.",
    "river.preprocessing.OneHotEncoder": "Categorical encoder, not an ML algorithm.",
    "river.preprocessing.OrdinalEncoder": "Categorical encoder, not an ML algorithm.",
    "river.preprocessing.PredClipper": "Prediction-clipping wrapper, not a standalone ML algorithm.",
    "river.preprocessing.PreviousImputer": "Missing-value imputer, not an ML algorithm.",
    "river.preprocessing.RobustScaler": "Feature scaler, not an ML algorithm.",
    "river.preprocessing.SparseRandomProjector": "Random projection transformer, not an ML algorithm.",
    "river.preprocessing.StandardScaler": "Feature scaler, not an ML algorithm.",
    "river.preprocessing.StatImputer": "Missing-value imputer, not an ML algorithm.",
    "river.preprocessing.TargetMinMaxScaler": "Target scaler wrapper, not a standalone ML algorithm.",
    "river.preprocessing.TargetStandardScaler": "Target scaler wrapper, not a standalone ML "
    "algorithm.",
}
