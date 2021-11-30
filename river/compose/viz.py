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

    details = ET.Element("details", attrib={"class": "component estimator"})

    summary = ET.Element("summary")
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "estimator-name"})
    pre.text = str(estimator)
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "estimator-params"})
    if isinstance(estimator, compose.FuncTransformer):
        code.text = f"\n{inspect.getsource(estimator.func)}\n"
    else:
        code.text = f"\n{pprint.pformat(estimator.__dict__)}\n\n"
    details.append(code)

    return details


def pipeline_to_html(pipeline) -> ET:

    div = ET.Element("div", attrib={"class": "component pipeline"})

    for step in pipeline.steps.values():
        div.append(to_html(step))

    return div


def union_to_html(union) -> ET:

    div = ET.Element("div", attrib={"class": "component union"})

    for transformer in union.transformers.values():
        div.append(to_html(transformer))

    return div


def wrapper_to_html(wrapper) -> ET:

    div = ET.Element("div", attrib={"class": "component wrapper"})

    details = ET.Element("details")
    div.append(details)

    summary = ET.Element("summary")
    details.append(summary)

    pre = ET.Element("pre", attrib={"class": "estimator-name"})
    pre.text = str(wrapper.__class__.__name__)
    summary.append(pre)

    code = ET.Element("code", attrib={"class": "estimator-params"})
    code.text = f"\n{pprint.pformat(wrapper.__dict__)}\n\n"
    details.append(code)

    div.append(to_html(wrapper._wrapped_model))

    return div


CSS = """
.estimator {
    padding: 1em;
    border-style: solid;
    background: white;
}

.pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(#000, #000) no-repeat center / 3px 100%;
}

.union {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white
}

.wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.wrapper > .estimator {
    margin-top: 1em;
}

/* Vertical spacing between steps */

.component + .component {
    margin-top: 2em;
}

.union > .estimator {
    margin-top: 0;
}

.union > .pipeline {
    margin-top: 0;
}

/* Spacing within a union of estimators */

.union > .component + .component {
    margin-left: 1em;
}

/* Typography */

.estimator-params {
    display: block;
    white-space: pre-wrap;
    font-size: 120%;
    margin-bottom: -1em;
}

.estimator > code,
.wrapper > details > code {
    background-color: white !important;
}

.estimator-name {
    display: inline;
    margin: 0;
    font-size: 130%;
}

/* Toggle */

summary {
    display: flex;
    align-items:center;
    cursor: pointer;
}

summary > div {
    width: 100%;
}
"""
