from vowpalwabbit import pyvw

from river import base


class VW2RiverBase:
    def __init__(self, *args, **kwargs):
        self.vw = pyvw.Workspace(*args, **kwargs)

    def _format_x(self, x):
        return " ".join(f"{k}:{v}" for k, v in x.items())


class VW2RiverClassifier(VW2RiverBase, base.Classifier):
    def learn_one(self, x, y):

        # Convert {False, True} to {-1, 1}
        y = int(y)
        y_vw = 2 * y - 1

        ex = self._format_x(x)
        ex = f"{y_vw} | {ex}"
        self.vw.learn(ex)
        return self

    def predict_proba_one(self, x):
        ex = "| " + self._format_x(x)
        y_pred = self.vw.predict(ex)
        return {True: y_pred, False: 1.0 - y_pred}
