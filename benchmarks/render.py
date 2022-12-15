import json
from typing import List

from dominate.tags import pre
from watermark import watermark
import pandas as pd


def render_df(df:pd.DataFrame, measures:List[str])-> dict:
    unique_datasets = list(df['dataset'].unique())
    res = {
        "data" : {"values": df.to_dict(orient="records")},
        "params": [
            {
                "name": "models",
                "select": {"type": "point", "fields": ["model"]},
                "bind": "legend"
            },
            {
                "name": "Dataset",
                "type": "single",
                "fields": ["dataset"],
                "value" : unique_datasets[0],
                "bind": {"input": "select", "options": unique_datasets}
            }
        ],
        "transform": [
            {"filter": {"field": "dataset", "equal": {"expr": "Dataset"}}}
        ],
        "repeat": {"row": measures},
        "spec" : {
            "width": "container",
            #"height": "container",
            "mark": "line",
            "encoding": {
                "x": {
                    "name": "Instance",
                    "field": "step",
                    "type": "quantitative",
                    "axis": {
                        "titleFontSize": 18,
                        "labelFontSize": 18,
                        "title": "Instance",
                    }
                },
                "y": {
                    "field": {"repeat": "row"},
                    "type": "quantitative",
                    "axis": {"titleFontSize": 18, "labelFontSize": 18}
                },
                "color": {
                    "field": "model",
                    "scale": {"scheme": "category20b"},
                    "title": "Models",
                    "legend": {
                        "titleFontSize": 18,
                        "labelFontSize": 18,
                        "labelLimit": 500,
                    },
                },
                "opacity": {
                    "condition": {"param": "models", "value": 1},
                    "value": 0.2
                }
            }
        }
    }
    return res


if __name__ == '__main__':

    with open("../docs/benchmarks/index.md", "w", encoding='utf-8') as f:
        print_ = lambda x: print(x, file=f, end="\n\n")
        print_(
    """---
hide:
- navigation
---
"""
        )

        print_("# Benchmarks")

        print_("## Binary Classification")
        print_("```vegalite")
        bin_clf_df = pd.read_csv("Binary classification.csv")
        measures = list(bin_clf_df.columns)[4:]
        print_(json.dumps(render_df(bin_clf_df, measures), indent=4))
        print_("```")

        print_("## Multi-Class Classification")
        print_("```vegalite")
        bin_clf_df = pd.read_csv("Multiclass classification.csv")
        measures = list(bin_clf_df.columns)[4:]
        print_(json.dumps(render_df(bin_clf_df, measures), indent=4))
        print_("```")

        print_("## Regression")
        print_("```vegalite")
        bin_clf_df = pd.read_csv("Regression.csv")
        measures = list(bin_clf_df.columns)[4:]
        print_(json.dumps(render_df(bin_clf_df, measures), indent=4))
        print_("```")

        print_("## Environment")
        print_(
            pre(watermark(python=True,
                          packages="river,numpy,scikit-learn,pandas,scipy",
                          machine=True))
        )