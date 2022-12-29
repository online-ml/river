import torch
class TorchMLPClassifier(torch.nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 5):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, hidden_size)
        self.nonlin = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.nonlin(self.linear1(x))
        x = self.nonlin(self.linear2(x))
        x = self.softmax(x)
        return x

class TorchMLPRegressor(torch.nn.Module):

    def __init__(self, n_features: int, hidden_size: int = 5):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, hidden_size)
        self.nonlin = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.nonlin(self.linear1(x))
        x = self.nonlin(self.linear2(x))
        return x

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X):
        X = self.linear(X)
        return self.softmax(X)

class TorchLinearRegression(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, X):
        return self.linear(X)
class TorchLSTMClassifier(torch.nn.Module):
    def __init__(self, n_features, num_layers=1, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=n_features, num_layers=num_layers, hidden_size=hidden_size, batch_first=False, bias=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        out, (hn, cn) = self.lstm(X)
        X = self.fc(out[-1, :])
        return self.softmax(X)

class TorchLSTMRegressor(torch.nn.Module):
    def __init__(self, n_features, num_layers=1, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=n_features, num_layers=num_layers, hidden_size=hidden_size, batch_first=False, bias=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, X, **kwargs):
        out, (hn, cn) = self.lstm(X)
        return self.fc(out[-1, :])