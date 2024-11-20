import math
from river import metrics, base, neighbors




class HoeffdingRace(base.Classifier):
    """


    >>> from river import model_selection
    >>> from river import linear_model, neighbors, tree, metrics, datasets

    >>> hoeffding_race = model_selection.HoeffdingRace(
    ...     models = {
    ...     "KNN": neighbors.KNNClassifier(),
    ...     "Log_Reg":linear_model.LogisticRegression()},
    ...     metric=metrics.Accuracy(),
    ...     delta=0.05
    ... )
    >>> dataset = datasets.Phishing()
    >>> for x, y in dataset:
    ...     hoeffding_race.learn_one(x, y)
    ...     if hoeffding_race.single_model_remaining():
    ...             break
    ...
    >>> hoeffding_race.remaining_models
    {'KNN'}
    
    """
    def __init__(self, models={"KNN":neighbors.KNNClassifier()}, delta=0.05, metric=metrics.Accuracy()):
       
        self.models = models
        self.delta = delta
        self.metric = metric  
        self.n = 0
        self.model_metrics = {name: metric.clone() for name in models.keys()}
        self.model_performance = {name: 0 for name in models.keys()}  
        self.remaining_models = set(models.keys()) 


    def hoeffding_bound(self, n):
        return math.sqrt((math.log(1 / self.delta)) / (2 * n))

    def learn_one(self, x, y):

        best_perf = max(self.model_performance.values()) if self.n > 0 else 0
        self.n = self.n+1

        for name in list(self.remaining_models):

            y_pred = self.models[name].predict_one(x)
            self.models[name].learn_one(x, y)
    
            # Update performance
            self.model_metrics[name].update(y, y_pred)
            self.model_performance[name] = self.model_metrics[name].get() 

            if self.model_performance[name]  + self.hoeffding_bound(self.n) < best_perf:
                self.remaining_models.remove(name)
                
    

    def predict_one(self, x):
        # Prediction by best remaining model
        if len(self.remaining_models) == 1:
            return self.models[list(self.remaining_models)[0]].predict_one(x)
        return None  # Pas de prédiction tant qu'un modèle n'est pas sélectionné
    
    def single_model_remaining(self):
        return len(self.remaining_models) == 1
    
    
