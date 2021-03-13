import pandas as pd
from rich.progress import Progress
from river.base import Classifier
from river.compat import SKL2RiverClassifier
from river.compose import Pipeline
from river.evaluate import load_binary_clf_tracks
from river.linear_model import LogisticRegression
from river.optim import SGD
from river.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import torch
from vowpalwabbit import pyvw


LEARNING_RATE = 0.005

class PyTorchLogReg(torch.nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class PyTorchModel:
    def __init__(self, network_func, loss, optimizer_func):
        self.network_func = network_func
        self.loss = loss
        self.optimizer_func = optimizer_func

        self.network = None
        self.optimizer = None

    def learn_one(self, x, y):

        # We only know how many features a dataset contains at runtime
        if self.network is None:
            self.network = self.network_func(n_features=len(x))
            self.optimizer = self.optimizer_func(self.network.parameters())

        x = torch.FloatTensor(list(x.values()))
        y = torch.FloatTensor([y])

        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self

class PyTorchBinaryClassifier(PyTorchModel, Classifier):

    def predict_proba_one(self, x):

        # We only know how many features a dataset contains at runtime
        if self.network is None:
            self.network = self.network_func(n_features=len(x))
            self.optimizer = self.optimizer_func(self.network.parameters())

        x = torch.FloatTensor(list(x.values()))
        p = self.network(x).item()
        return {True: p, False: 1.0 - p}

class VW2RiverBase:

    def __init__(self, *args, **kwargs):
        self.vw = pyvw.vw(*args, **kwargs)

    def _format_x(self, x):
        return ' '.join(f'{k}:{v}' for k, v in x.items())


class VW2RiverClassifier(VW2RiverBase, Classifier):

    def learn_one(self, x, y):

        # Convert {False, True} to {-1, 1}
        y = int(y)
        y_vw = 2 * y - 1

        ex = self._format_x(x)
        ex = f'{y_vw} | {ex}'
        self.vw.learn(ex)
        return self

    def predict_proba_one(self, x):
        ex = '| ' + self._format_x(x)
        y_pred = self.vw.predict(ex)
        return {True: y_pred, False: 1.0 - y_pred}

MODELS = {
    'River': Pipeline(
        StandardScaler(),
        LogisticRegression(optimizer=SGD(LEARNING_RATE))
    ),
    'scikit-learn': Pipeline(
        StandardScaler(),
        SKL2RiverClassifier(
            SGDClassifier(
                loss='log',
                learning_rate='constant',
                eta0=LEARNING_RATE,
                penalty='none'
            ),
            classes=[False, True]
        )
    ),
    'PyTorch': PyTorchBinaryClassifier(
        network_func=PyTorchLogReg,
        loss=torch.nn.BCELoss(),
        optimizer_func=lambda params: torch.optim.SGD(params, lr=LEARNING_RATE)
    ),
    'Vowpal Wabbit': VW2RiverClassifier(
        sgd=True,
        learning_rate=LEARNING_RATE,
        loss_function='logistic',
        link='logistic',
        adaptive=False,
        normalized=False,
        invariant=False,
        l2=0.,
        l1=0.,
        power_t=0,
        quiet=True
    )
}

def run():

    results = []
    tracks = load_binary_clf_tracks()
    n_checkpoints = 10

    with Progress() as progress:
        bar = progress.add_task("Models", total=len(MODELS))

        for model_name, model in MODELS.items():
            model_bar = progress.add_task(f"[green]{model_name}", total=len(tracks))

            for track in tracks:
                track_bar = progress.add_task(f"[cyan]{track.name}", total=n_checkpoints)

                for step in track.run(model, n_checkpoints=n_checkpoints):
                    step['Model'] = model_name
                    step['Track'] = track.name
                    results.append(step)
                    progress.advance(track_bar)
                progress.advance(model_bar)
            progress.advance(bar)

    return pd.DataFrame(results).set_index(['Model', 'Track', 'Step']).reset_index()



if __name__ == '__main__':
    results = run()

    with open('README.md', 'w') as f:
        print('# Logistic regression\n', file=f)

        print('## Final results\n', file=f)
        final = (
            results
            .sort_values('Step')
            .groupby(['Model', 'Track'])
            .last()
            .reset_index()
            .drop(columns=['Step'])
        )
        print(final.to_markdown(index=False), file=f)
        print('', file=f)

        print('## Traces\n', file=f)
        print(results.to_markdown(index=False), file=f)
