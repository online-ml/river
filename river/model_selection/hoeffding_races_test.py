from river import linear_model, neighbors, tree
from river.datasets import synth
from river import datasets
from river import metrics
from hoeffding_races import HoeffdingRace

metric = metrics.Accuracy()

# Modèles candidats
models = {
    'Regression': linear_model.LogisticRegression(),
    'KNN': neighbors.KNNClassifier(),
    'DecisionTree': tree.HoeffdingTreeClassifier()
}

# Initialisation de HoeffdingRace

hoeffding_race = HoeffdingRace(models=models,metric=metric)

# Exécution sur un flux de données
dataset = datasets.CreditCard()
n=0
for x,y in dataset:
    selected_model = hoeffding_race.learn_one(x, y)
    if selected_model:
        break
print(n)
print(hoeffding_race.model_performance)