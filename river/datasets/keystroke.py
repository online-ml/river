from __future__ import annotations

from river import stream

from . import base


class Keystroke(base.RemoteDataset):
    """CMU keystroke dataset.

    Users are tasked to type in a password. The task is to determine which user is typing in the
    password.

    The only difference with the original dataset is that the "sessionIndex" and "rep" attributes
    have been dropped.

    References
    ----------
    [^1]: [Keystroke Dynamics - Benchmark Data Set](http://www.cs.cmu.edu/~keystroke/)

    """

    def __init__(self):
        super().__init__(
            n_samples=20_400,
            n_features=31,
            n_classes=51,
            task=base.MULTI_CLF,
            url="http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv",
            size=4_669_935,
            filename="DSL-StrongPasswordData.csv",
            unpack=False,
        )

    def _iter(self):
        converters = {
            "H.period": float,
            "DD.period.t": float,
            "UD.period.t": float,
            "H.t": float,
            "DD.t.i": float,
            "UD.t.i": float,
            "H.i": float,
            "DD.i.e": float,
            "UD.i.e": float,
            "H.e": float,
            "DD.e.five": float,
            "UD.e.five": float,
            "H.five": float,
            "DD.five.Shift.r": float,
            "UD.five.Shift.r": float,
            "H.Shift.r": float,
            "DD.Shift.r.o": float,
            "UD.Shift.r.o": float,
            "H.o": float,
            "DD.o.a": float,
            "UD.o.a": float,
            "H.a": float,
            "DD.a.n": float,
            "UD.a.n": float,
            "H.n": float,
            "DD.n.l": float,
            "UD.n.l": float,
            "H.l": float,
            "DD.l.Return": float,
            "UD.l.Return": float,
            "H.Return": float,
        }
        return stream.iter_csv(
            self.path,
            target="subject",
            converters=converters,
            drop=["sessionIndex", "rep"],
        )
