from __future__ import annotations

from river import stream

from . import base


class Music(base.RemoteDataset):
    """Multi-label music mood prediction.

    The goal is to predict to which kinds of moods a song pertains to.

    References
    ----------
    [^1]: [Read, J., Reutemann, P., Pfahringer, B. and Holmes, G., 2016. MEKA: a multi-label/multi-target extension to WEKA. The Journal of Machine Learning Research, 17(1), pp.667-671.](http://www.jmlr.org/papers/v17/12-164.html)

    """

    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=593,
            n_features=72,
            n_outputs=6,
            url="https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/music.csv",
            size=378_980,
            unpack=False,
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target=[
                "amazed-suprised",
                "happy-pleased",
                "relaxing-clam",
                "quiet-still",
                "sad-lonely",
                "angry-aggresive",
            ],
            converters={
                "amazed-suprised": lambda x: x == "1",
                "happy-pleased": lambda x: x == "1",
                "relaxing-clam": lambda x: x == "1",
                "quiet-still": lambda x: x == "1",
                "sad-lonely": lambda x: x == "1",
                "angry-aggresive": lambda x: x == "1",
                "Mean_Acc1298_Mean_Mem40_Centroid": float,
                "Mean_Acc1298_Mean_Mem40_Rolloff": float,
                "Mean_Acc1298_Mean_Mem40_Flux": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_0": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_1": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_2": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_3": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_4": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_5": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_6": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_7": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_8": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_9": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_10": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_11": float,
                "Mean_Acc1298_Mean_Mem40_MFCC_12": float,
                "Mean_Acc1298_Std_Mem40_Centroid": float,
                "Mean_Acc1298_Std_Mem40_Rolloff": float,
                "Mean_Acc1298_Std_Mem40_Flux": float,
                "Mean_Acc1298_Std_Mem40_MFCC_0": float,
                "Mean_Acc1298_Std_Mem40_MFCC_1": float,
                "Mean_Acc1298_Std_Mem40_MFCC_2": float,
                "Mean_Acc1298_Std_Mem40_MFCC_3": float,
                "Mean_Acc1298_Std_Mem40_MFCC_4": float,
                "Mean_Acc1298_Std_Mem40_MFCC_5": float,
                "Mean_Acc1298_Std_Mem40_MFCC_6": float,
                "Mean_Acc1298_Std_Mem40_MFCC_7": float,
                "Mean_Acc1298_Std_Mem40_MFCC_8": float,
                "Mean_Acc1298_Std_Mem40_MFCC_9": float,
                "Mean_Acc1298_Std_Mem40_MFCC_10": float,
                "Mean_Acc1298_Std_Mem40_MFCC_11": float,
                "Mean_Acc1298_Std_Mem40_MFCC_12": float,
                "Std_Acc1298_Mean_Mem40_Centroid": float,
                "Std_Acc1298_Mean_Mem40_Rolloff": float,
                "Std_Acc1298_Mean_Mem40_Flux": float,
                "Std_Acc1298_Mean_Mem40_MFCC_0": float,
                "Std_Acc1298_Mean_Mem40_MFCC_1": float,
                "Std_Acc1298_Mean_Mem40_MFCC_2": float,
                "Std_Acc1298_Mean_Mem40_MFCC_3": float,
                "Std_Acc1298_Mean_Mem40_MFCC_4": float,
                "Std_Acc1298_Mean_Mem40_MFCC_5": float,
                "Std_Acc1298_Mean_Mem40_MFCC_6": float,
                "Std_Acc1298_Mean_Mem40_MFCC_7": float,
                "Std_Acc1298_Mean_Mem40_MFCC_8": float,
                "Std_Acc1298_Mean_Mem40_MFCC_9": float,
                "Std_Acc1298_Mean_Mem40_MFCC_10": float,
                "Std_Acc1298_Mean_Mem40_MFCC_11": float,
                "Std_Acc1298_Mean_Mem40_MFCC_12": float,
                "Std_Acc1298_Std_Mem40_Centroid": float,
                "Std_Acc1298_Std_Mem40_Rolloff": float,
                "Std_Acc1298_Std_Mem40_Flux": float,
                "Std_Acc1298_Std_Mem40_MFCC_0": float,
                "Std_Acc1298_Std_Mem40_MFCC_1": float,
                "Std_Acc1298_Std_Mem40_MFCC_2": float,
                "Std_Acc1298_Std_Mem40_MFCC_3": float,
                "Std_Acc1298_Std_Mem40_MFCC_4": float,
                "Std_Acc1298_Std_Mem40_MFCC_5": float,
                "Std_Acc1298_Std_Mem40_MFCC_6": float,
                "Std_Acc1298_Std_Mem40_MFCC_7": float,
                "Std_Acc1298_Std_Mem40_MFCC_8": float,
                "Std_Acc1298_Std_Mem40_MFCC_9": float,
                "Std_Acc1298_Std_Mem40_MFCC_10": float,
                "Std_Acc1298_Std_Mem40_MFCC_11": float,
                "Std_Acc1298_Std_Mem40_MFCC_12": float,
                "BH_LowPeakAmp": float,
                "BH_LowPeakBPM": int,
                "BH_HighPeakAmp": float,
                "BH_HighPeakBPM": int,
                "BH_HighLowRatio": int,
                "BHSUM1": float,
                "BHSUM2": float,
                "BHSUM3": float,
            },
        )
