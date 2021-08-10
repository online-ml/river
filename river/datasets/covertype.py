from river import stream
from river.datasets import base


class Covertype(base.RemoteDataset):
    """Covertype Dataset.

        Normalized version of the Forest Covertype dataset (see version 1), so that the numerical values are between 0 and 1. Contains the forest cover type for 30 x 30 meter cells obtained from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. It contains 581,012 instances and 54 attributes, and it has been used in several papers on data stream classification.

        Parameters
        ----------

        Examples
        --------

        >>> from river import synth

        >>> dataset = synth.Covertype()

        >>> for x, y in dataset.take(5):
        ...     print(list(x.values()), y)
        [0.368684, 0.141667, 0.045455, 0.184681, 0.223514, 0.071659, 0.870079, 0.913386, 0.582677, 0.875366, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 5
        [0.365683, 0.155556, 0.030303, 0.151754, 0.215762, 0.054798, 0.866142, 0.925197, 0.594488, 0.867838, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 5
        [0.472736, 0.386111, 0.136364, 0.19184, 0.307494, 0.446817, 0.92126, 0.937008, 0.531496, 0.853339, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 2
        [0.463232, 0.430556, 0.272727, 0.173228, 0.375969, 0.434172, 0.937008, 0.937008, 0.480315, 0.865886, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 2
        [0.368184, 0.125, 0.030303, 0.10952, 0.222222, 0.054939, 0.866142, 0.92126, 0.590551, 0.860449, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 5

        References
        ----------
        [^1]: [MOA](https://moa.cms.waikato.ac.nz/datasets/)

        """

    def __init__(self):
        super().__init__(
            url="https://datahub.io/machine-learning/covertype/r/covertype.csv",
            size=103930879,
            unpack=False,
            task=base.MULTI_CLF,
            n_samples=58_1012,
            n_features=54,
            filename="covertype.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="class",
            converters={
                'Elevation': float,
                'Aspect': float,
                'Slope': float,
                'Horizontal_Distance_To_Hydrology': float,
                'Vertical_Distance_To_Hydrology': float,
                'Horizontal_Distance_To_Roadways': float,
                'Hillshade_9am': float,
                'Hillshade_Noon': float,
                'Hillshade_3pm': float,
                'Horizontal_Distance_To_Fire_Points': float,
                'Wilderness_Area1': int,
                'Wilderness_Area2': int,
                'Wilderness_Area3': int,
                'Wilderness_Area4': int,
                'Soil_Type1': int,
                'Soil_Type2': int,
                'Soil_Type3': int,
                'Soil_Type4': int,
                'Soil_Type5': int,
                'Soil_Type6': int,
                'Soil_Type7': int,
                'Soil_Type8': int,
                'Soil_Type9': int,
                'Soil_Type10': int,
                'Soil_Type11': int,
                'Soil_Type12': int,
                'Soil_Type13': int,
                'Soil_Type14': int,
                'Soil_Type15': int,
                'Soil_Type16': int,
                'Soil_Type17': int,
                'Soil_Type18': int,
                'Soil_Type19': int,
                'Soil_Type20': int,
                'Soil_Type21': int,
                'Soil_Type22': int,
                'Soil_Type23': int,
                'Soil_Type24': int,
                'Soil_Type25': int,
                'Soil_Type26': int,
                'Soil_Type27': int,
                'Soil_Type28': int,
                'Soil_Type29': int,
                'Soil_Type30': int,
                'Soil_Type31': int,
                'Soil_Type32': int,
                'Soil_Type33': int,
                'Soil_Type34': int,
                'Soil_Type35': int,
                'Soil_Type36': int,
                'Soil_Type37': int,
                'Soil_Type38': int,
                'Soil_Type39': int,
                'Soil_Type40' :int,
        }
        )