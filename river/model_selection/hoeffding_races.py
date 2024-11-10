import math
from river import metrics, base


class HoeffdingRace(base.Estimator):
    def __init__(self, models, delta=0.05, metric=metrics.Accuracy()):
        # Initialisation des modèles, du delta, de la métrique par défaut et des variables de suivi
        self.models = models
        self.delta = delta
        self.metric = metric  # Métrique de base (peut être utilisée pour chaque modèle)
        self.n_obs = 0  # Compteur
        # Initialiser des métriques distinctes pour chaque modèle
        self.model_metrics = {name: metric.clone() for name in models.keys()}
        self.model_performance = {name: 0 for name in models.keys()}  # Suivi des performances des modèles
        self.remaining_models = set(models.keys())  # Liste des modèles restant


    def hoeffding_bound(self, n):
        return math.sqrt((math.log(1 / self.delta)) / (2 * n))

    def learn_one(self, x, y):
        self.n_obs += 1
        best_perf = max(self.model_performance.values()) / self.n_obs if self.n_obs > 0 else 0

        for name in list(self.remaining_models):
            y_pred = self.models[name].predict_one(x)
            
            self.models[name].learn_one(x, y)
            
            # Update performance
            result = self.model_metrics[name].update(y, y_pred)
            
            self.model_performance[name] += self.model_metrics[name].get() / self.n_obs 
            
            # Elimination check
            avg_perf = self.model_performance[name] 

            if avg_perf + self.hoeffding_bound(self.n_obs) < best_perf:
                self.remaining_models.remove(name)
                print(f"{name} éliminé après {self.n_obs} observations")
            

        
        # Early stopping if only one model remains
        if len(self.remaining_models) == 1:
            print(f"{list(self.remaining_models)[0]} est le modèle sélectionné.")
            return list(self.remaining_models)[0]
       
    

    def predict_one(self, x):
        # Prediction by best remaining model
        if len(self.remaining_models) == 1:
            return self.models[list(self.remaining_models)[0]].predict_one(x)
        return None  # Pas de prédiction tant qu'un modèle n'est pas sélectionné
    
