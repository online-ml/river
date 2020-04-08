import numbers
import typing

RegTarget = numbers.Number
ClfTarget = typing.Union[bool, str, int]
Target = typing.Union[ClfTarget, RegTarget]
Stream = typing.Iterator[typing.Tuple[dict, typing.Optional[Target]]]
