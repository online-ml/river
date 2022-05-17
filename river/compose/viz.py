import inspect
import pprint
from xml.etree import ElementTree as ET

from river import base, compose


def to_html(obj) -> ET:
    if isinstance(obj, compose.Pipeline):
        return pipeline_to_html(obj)
    if isinstance(obj, compose.TransformerUnion):
        return union_to_html(obj)
    if isinstance(obj, base.Wrapper):
        return wrapper_to_html(obj)
    return estimator_to_html(obj)


def estimator_to_html(estimator) -> ET:

    details = ET.Element("details", attrib={"class": "river-component river-estimator"})

    summary = ET.Element("summary", attrib={"class": "river-summary"})
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "river-estimator-name"})
    pre.text = str(estimator)
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "river-estimator-params"})
    if isinstance(estimator, compose.FuncTransformer):
        code.text = f"\n{inspect.getsource(estimator.func)}\n"
    else:
        code.text = f"\n{pprint.pformat(estimator.__dict__)}\n\n"
    details.append(code)

    return details


def pipeline_to_html(pipeline) -> ET:

    div = ET.Element("div", attrib={"class": "river-component river-pipeline"})

    for step in pipeline.steps.values():
        div.append(to_html(step))

    return div


def union_to_html(union) -> ET:

    div = ET.Element("div", attrib={"class": "river-component river-union"})

    for transformer in union.transformers.values():
        div.append(to_html(transformer))

    return div


def wrapper_to_html(wrapper) -> ET:

    div = ET.Element("div", attrib={"class": "river-component river-wrapper"})

    details = ET.Element("details", attrib={"class": "river-details"})
    div.append(details)

    summary = ET.Element("summary", attrib={"class": "river-summary"})
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "river-estimator-name"})
    pre.text = str(wrapper.__class__.__name__)
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "river-estimator-params"})
    code.text = f"\n{pprint.pformat(wrapper.__dict__)}\n\n"
    details.append(code)

    div.append(to_html(wrapper._wrapped_model))

    return div


CSS = """
.river-estimator {
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(#000, #000) no-repeat center / 3px 100%;
}

.river-union {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white
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
    font-size: 120%;
    margin-bottom: -1em;
}

.river-estimator > .river-estimator-params,
.river-wrapper > .river-details > river-estimator-params {
    background-color: white !important;
}

.river-estimator-name {
    display: inline;
    margin: 0;
    font-size: 130%;
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
