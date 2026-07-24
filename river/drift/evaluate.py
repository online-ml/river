from __future__ import annotations

import collections
import math
import typing

__all__ = [
    "Matching",
    "detection_delay",
    "episode_recall",
    "f1",
    "false_alarm_rate",
    "match",
    "mean_time_between_false_alarms",
    "mean_time_ratio",
    "missed_detection_rate",
    "normalized_detection_time",
    "precision",
    "recall",
    "report",
    "score",
]

Drift = int | tuple[int, int]

Matching = collections.namedtuple("Matching", ["hits", "false_alarms"])


def _windows(drifts: list[Drift], delta_max: int, delta_pre: int) -> list[tuple[int, int, int]]:
    if delta_max <= 0:
        raise ValueError("delta_max must be strictly positive")
    if delta_pre < 0:
        raise ValueError("delta_pre must be positive")

    windows = []
    for drift in drifts:
        start, end = drift if isinstance(drift, tuple) else (drift, drift)
        if end < start:
            raise ValueError(f"drift {drift} ends before it starts")
        windows.append((start, start - delta_pre, end + delta_max))
    return windows


def match(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> Matching:
    """Assign each alarm to a drift, or to the false alarm pile.

    A drift is an index, or a `(start, end)` pair for a drift that spans several samples. Its
    acceptable detection window is `[start - delta_pre, end + delta_max]`, and both bounds are
    inclusive. An alarm inside a window is a true positive; an alarm inside none of them is a false
    alarm. Alarms are matched to the earliest window they fall into.

    Parameters
    ----------
    alarms
        The indices at which the detector fired.
    drifts
        The ground truth drifts, in chronological order.
    delta_max
        How late a detection may be and still count.
    delta_pre
        How early a detection may be and still count.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.match(alarms=[110, 900], drifts=[100], delta_max=50)
    Matching(hits={0: [110]}, false_alarms=[900])

    """
    windows = _windows(list(drifts), delta_max, delta_pre)

    hits: dict[int, list[int]] = {}
    false_alarms = []

    for alarm in alarms:
        for i, (_, lower, upper) in enumerate(windows):
            if lower <= alarm <= upper:
                hits.setdefault(i, []).append(alarm)
                break
        else:
            false_alarms.append(alarm)

    return Matching(hits=hits, false_alarms=false_alarms)


def precision(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The share of alarms that landed inside a detection window.

    Undefined, and therefore `nan`, when the detector never fired.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.precision(alarms=[110, 900], drifts=[100], delta_max=50)
    0.5

    """
    alarms = list(alarms)
    if not alarms:
        return math.nan

    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    n_hits = sum(map(len, matching.hits.values()))
    return n_hits / len(alarms)


def recall(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The share of true positives among true positives and missed drifts.

    Every alarm inside a window counts, so a detector that fires repeatedly inside one window is
    rewarded for it. Use `episode_recall` to count each window once.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.recall(alarms=[110, 130], drifts=[100, 500], delta_max=50)
    0.666666...

    """
    drifts = list(drifts)
    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)

    n_hits = sum(map(len, matching.hits.values()))
    n_missed = len(drifts) - len(matching.hits)

    if n_hits + n_missed == 0:
        return math.nan
    return n_hits / (n_hits + n_missed)


def episode_recall(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The share of drifts that got at least one alarm.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.episode_recall(alarms=[110, 130], drifts=[100, 500], delta_max=50)
    0.5

    """
    drifts = list(drifts)
    if not drifts:
        return math.nan

    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    return len(matching.hits) / len(drifts)


def missed_detection_rate(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The share of drifts that went unnoticed.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.missed_detection_rate(alarms=[110], drifts=[100, 500], delta_max=50)
    0.5

    """
    caught = episode_recall(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    if math.isnan(caught):
        return math.nan
    return 1 - caught


def f1(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The harmonic mean of precision and episode recall.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.f1(alarms=[110, 900], drifts=[100, 500], delta_max=50)
    0.5

    """
    alarms = list(alarms)
    drifts = list(drifts)

    p = precision(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    r = episode_recall(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)

    if math.isnan(p) or math.isnan(r):
        return math.nan
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def detection_delay(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """How long the first alarm took, averaged over the drifts that were caught.

    Missed drifts are left out rather than charged a penalty, so this has to be read next to
    `missed_detection_rate`. Undefined, and therefore `nan`, when nothing was caught.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.detection_delay(alarms=[110, 530], drifts=[100, 500, 900], delta_max=50)
    20.0

    """
    drifts = list(drifts)
    windows = _windows(drifts, delta_max, delta_pre)
    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)

    delays = [min(hit) - windows[i][0] for i, hit in matching.hits.items()]
    if not delays:
        return math.nan
    return sum(delays) / len(delays)


def normalized_detection_time(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The detection delay as a fraction of `delta_max`.

    This is what makes delays comparable between streams whose drifts are spaced differently.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.normalized_detection_time(alarms=[110], drifts=[100], delta_max=50)
    0.2

    """
    return detection_delay(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre) / delta_max


def false_alarm_rate(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    n_samples: int,
    delta_max: int,
    delta_pre: int = 0,
    unit: int = 1,
) -> float:
    """How many false alarms were raised per `unit` samples.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.false_alarm_rate(
    ...     alarms=[110, 900], drifts=[100], n_samples=1000, delta_max=50, unit=1000
    ... )
    1.0

    """
    if n_samples <= 0:
        raise ValueError("n_samples must be strictly positive")

    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    return len(matching.false_alarms) / n_samples * unit


def mean_time_between_false_alarms(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """The average gap between consecutive false alarms.

    Undefined, and therefore `nan`, below two false alarms, because there is no gap to measure.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.mean_time_between_false_alarms(
    ...     alarms=[200, 300, 500], drifts=[], delta_max=50
    ... )
    150.0

    """
    matching = match(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)

    false_alarms = matching.false_alarms
    if len(false_alarms) < 2:
        return math.nan

    gaps = [later - earlier for earlier, later in zip(false_alarms, false_alarms[1:])]
    return sum(gaps) / len(gaps)


def mean_time_ratio(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
) -> float:
    """A single score trading the three ground truth metrics off against each other.

    It is the mean time between false alarms, divided by the detection delay, discounted by the
    share of drifts that were missed. Higher is better. It inherits the domain of its parts, so it
    is `nan` below two false alarms or when nothing was caught.

    Examples
    --------

    >>> from river import drift

    >>> drift.evaluate.mean_time_ratio(alarms=[110, 400, 700], drifts=[100, 900], delta_max=50)
    15.0

    """
    alarms = list(alarms)
    drifts = list(drifts)

    mtfa = mean_time_between_false_alarms(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    if math.isnan(mtfa):
        return math.nan

    mtd = detection_delay(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    if math.isnan(mtd):
        return math.nan
    if mtd == 0:
        return math.inf

    mdr = missed_detection_rate(alarms, drifts, delta_max=delta_max, delta_pre=delta_pre)
    return mtfa / mtd * (1 - mdr)


def report(
    alarms: typing.Iterable[int],
    drifts: typing.Iterable[Drift],
    *,
    n_samples: int,
    delta_max: int,
    delta_pre: int = 0,
    unit: int = 1,
) -> dict:
    """Every metric in this module, for one detector on one stream.

    Examples
    --------

    >>> from river import drift

    >>> report = drift.evaluate.report(
    ...     alarms=[110, 900], drifts=[100, 500], n_samples=1000, delta_max=50
    ... )
    >>> report["episode_recall"]
    0.5
    >>> report["detection_delay"]
    10.0

    """
    alarms = list(alarms)
    drifts = list(drifts)
    kwargs = {"delta_max": delta_max, "delta_pre": delta_pre}

    return {
        "n_samples": n_samples,
        "n_drifts": len(drifts),
        "n_alarms": len(alarms),
        "precision": precision(alarms, drifts, **kwargs),
        "recall": recall(alarms, drifts, **kwargs),
        "episode_recall": episode_recall(alarms, drifts, **kwargs),
        "f1": f1(alarms, drifts, **kwargs),
        "missed_detection_rate": missed_detection_rate(alarms, drifts, **kwargs),
        "detection_delay": detection_delay(alarms, drifts, **kwargs),
        "normalized_detection_time": normalized_detection_time(alarms, drifts, **kwargs),
        "false_alarm_rate": false_alarm_rate(
            alarms, drifts, n_samples=n_samples, unit=unit, **kwargs
        ),
        "mean_time_between_false_alarms": mean_time_between_false_alarms(alarms, drifts, **kwargs),
        "mean_time_ratio": mean_time_ratio(alarms, drifts, **kwargs),
    }


def score(
    detector,
    stream: typing.Iterable[typing.Any],
    drifts: typing.Iterable[Drift],
    *,
    delta_max: int,
    delta_pre: int = 0,
    unit: int = 1,
) -> dict:
    """Run a detector over a stream whose drifts are known, and report on it.

    Parameters
    ----------
    detector
        Anything with an `update` method and a `drift_detected` property.
    stream
        The values to feed the detector, one at a time.
    drifts
        The ground truth drifts, in chronological order.
    delta_max
        How late a detection may be and still count.
    delta_pre
        How early a detection may be and still count.
    unit
        The number of samples the false alarm rate is quoted per.

    Examples
    --------

    >>> import itertools
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)
    >>> stream = itertools.chain(
    ...     (rng.gauss(0, 0.1) for _ in range(1000)),
    ...     (rng.gauss(5, 0.1) for _ in range(1000)),
    ... )

    >>> report = drift.evaluate.score(drift.ADWIN(), stream, drifts=[1000], delta_max=200)
    >>> report["episode_recall"]
    1.0

    """
    alarms = []
    n_samples = 0

    for i, x in enumerate(stream):
        detector.update(x)
        if detector.drift_detected:
            alarms.append(i)
        n_samples = i + 1

    return report(
        alarms,
        drifts,
        n_samples=n_samples,
        delta_max=delta_max,
        delta_pre=delta_pre,
        unit=unit,
    )
