from __future__ import annotations

import inspect
import textwrap
from xml.etree import ElementTree as ET


def to_html(obj) -> ET.Element:
    from river import base, compose

    if isinstance(obj, compose.Pipeline):
        return pipeline_to_html(obj)
    if isinstance(obj, compose.TransformerUnion):
        return union_to_html(obj)
    if isinstance(obj, base.Wrapper):
        return wrapper_to_html(obj)
    return estimator_to_html(obj)


def estimator_to_html(estimator) -> ET.Element:
    from river import compose

    details = ET.Element("details", attrib={"class": "river-component river-estimator"})

    summary = ET.Element("summary", attrib={"class": "river-summary"})
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "river-estimator-name"})
    str_estimator = str(estimator)
    parts = str_estimator.split(" ")
    pre.text = (
        textwrap.shorten(str_estimator, width=20)
        if len(parts[0]) < 20
        else (parts[0] + (" [...]" if len(parts) > 1 else ""))
    )
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "river-estimator-params"})
    if isinstance(estimator, compose.FuncTransformer):
        code.text = f"\n{inspect.getsource(estimator.func)}\n"
    else:
        code.text = f"{repr(estimator)}\n"
    details.append(code)

    return details


def pipeline_to_html(pipeline) -> ET.Element:
    div = ET.Element("div", attrib={"class": "river-component river-pipeline"})

    for step in pipeline.steps.values():
        div.append(to_html(step))

    return div


def union_to_html(union) -> ET.Element:
    div = ET.Element("div", attrib={"class": "river-component river-union"})

    for transformer in union.transformers.values():
        div.append(to_html(transformer))

    return div


def wrapper_to_html(wrapper) -> ET.Element:
    div = ET.Element("div", attrib={"class": "river-component river-wrapper"})

    details = ET.Element("details", attrib={"class": "river-details"})
    div.append(details)

    summary = ET.Element("summary", attrib={"class": "river-summary"})
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "river-estimator-name"})
    pre.text = str(wrapper.__class__.__name__)
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "river-estimator-params"})
    code.text = f"{repr(wrapper)}\n"
    details.append(code)

    div.append(to_html(wrapper._wrapped_model))

    return div


CSS = """
.river-estimator {
    padding: 1em;
    border-style: solid;
    background: white;
    max-width: max-content;
}

.river-pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(#000, #000) no-repeat center / 1.5px 100%;
}

.river-union {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-wrapper > .river-estimator {
    margin-top: 1em;
}

/* Vertical spacing between steps */

.river-component + .river-component {
    margin-top: 2em;
}

.river-union > .river-estimator {
    margin-top: 0;
}

.river-union > .river-component {
    margin-top: 0;
}

.river-union > .pipeline {
    margin-top: 0;
}

/* Spacing within a union of estimators */

.river-union > .river-component + .river-component {
    margin-left: 1em;
}

/* Typography */

.river-estimator-params {
    display: block;
    white-space: pre-wrap;
    font-size: 110%;
    margin-top: 1em;
}

.river-estimator > .river-estimator-params,
.river-wrapper > .river-details > river-estimator-params {
    background-color: white !important;
}

.river-wrapper > .river-details {
    margin-bottom: 1em;
}

.river-estimator-name {
    display: inline;
    margin: 0;
    font-size: 110%;
}

/* Toggle */

.river-summary {
    display: flex;
    align-items:center;
    cursor: pointer;
}

.river-summary > div {
    width: 100%;
}
"""
