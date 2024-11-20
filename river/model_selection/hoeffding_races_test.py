
from hoeffding_races import HoeffdingRace
from river.linear_model import LogisticRegression
from river.metrics import Accuracy
from river import linear_model, neighbors, tree, metrics, datasets
from river import naive_bayes



# Instantiate a HoeffdingRace object with a single candidate model
hoeffding_race = HoeffdingRace(
    models = {"LogisticRegression": linear_model.LogisticRegression()},
    metric=Accuracy(),
    delta=0.05
)

dataset = datasets.AirlinePassengers()
print(dataset)
for x, y in dataset:
    hoeffding_race.learn_one(x, y)
    #print(hoeffding_race.model_metrics["KNN"].get())
    if len(hoeffding_race.remaining_models) == 1:
        break
print(hoeffding_race.model_performance)