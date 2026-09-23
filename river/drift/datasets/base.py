from __future__ import annotations

import abc
import pathlib
import typing

from river.datasets import base


class ChangePointFileDataset(base.FileDataset, abc.ABC):
    """Base class for change point datasets that are stored in a local file.

    Datasets that are part of the Alan Turing Institute Change Point Detection project.

    Parameters
    ----------
    annotations
        The file's associated annotations.
    desc
        Extra dataset parameters to pass as keyword arguments.

    """

    def __init__(
        self,
        annotations: dict[str, list[int]],
        **desc: typing.Any,
    ) -> None:
        super().__init__(**desc, directory=pathlib.Path(__file__).parent)  # type: ignore[no-untyped-call]
        self.annotations = annotations

    @property
    def _repr_content(self) -> dict[str, str]:
        repr_content: dict[str, str] = super()._repr_content
        repr_content["Annotators"] = f"{len(self.annotations):,d}"
        repr_content["Annotations"] = f"{sum(map(len, self.annotations.values())):,d}"
        return repr_content

    def _annotations_aggregated(self, annotator_aggregation: str) -> set[int] | dict[int, int]:
        """The function `annotations_aggregated` takes an annotator aggregation method as input and returns
        the aggregated annotations based on that method.

        Parameters
        ----------
        annotator_aggregation
            The parameter "annotator_aggregation" is a string that specifies the method of aggregating
        annotations from different annotators. It can take one of the following values:

        Returns
        -------
            the aggregated annotations based on the specified annotator aggregation method.

        """
        if annotator_aggregation == "union":
            annotations: set[int] = set()
            for annotator in self.annotations:
                annotations.update(self.annotations[annotator])
        elif annotator_aggregation == "intersection":
            annotations = set(self.annotations["0"])
            for annotator in self.annotations:
                annotations.intersection_update(self.annotations[annotator])
        elif annotator_aggregation == "majority":
            counts: dict[int, int] = {}
            for annotator in self.annotations:
                for change_point in self.annotations[annotator]:
                    if change_point in counts:
                        counts[change_point] += 1
                    else:
                        counts[change_point] = 1
            annotations = {
                change_point
                for change_point, count in counts.items()
                if count > len(self.annotations) / 2
            }
        else:
            raise ValueError("Unknown annotator aggregation method.")
        return annotations
