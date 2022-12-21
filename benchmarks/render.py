import json
from typing import List

from dominate.tags import pre
from watermark import watermark
import pandas as pd


def render_df(df:pd.DataFrame)-> dict:
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'Time in s'}, inplace=True)
    if 'Memory' in df.columns:
        df.rename(columns={'Memory': 'Memory in MB'}, inplace=True)
    unique_datasets = list(df['dataset'].unique())
    measures = list(df.columns)[4:]
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

    details = json.load(open('details.json'))

    with open("../docs/benchmarks/index.md", "w", encoding='utf-8') as f:
        print_ = lambda x: print(x, file=f, end="\n\n")
        print_(
    """---
hide:
- navigation
---
"""
        )

        print_('# Benchmark')

        for track_name, track_details in details.items():
            print_(f'## {track_name}')


            df = pd.read_csv(f'{track_name}.csv')
            print_("```vegalite")
            print_(json.dumps(render_df(df=df), indent=2))
            print_("```")

            print_('### Datasets')
            for dataset_name, dataset_details in track_details['Dataset'].items():
                print_(f'<details>')
                print_(f'<summary>{dataset_name}</summary>')
                print_(pre(dataset_details))
                print_(f'</details>')
            print_('### Models')
            for model_name, model_details in track_details['Model'].items():
                print_(f'<details>')
                print_(f'<summary>{model_name}</summary>')
                print_(pre(model_details))
                print_(f'</details>')

        print_("# Environment")
        print_(
            pre(watermark(python=True,
                          packages="river,numpy,scikit-learn,pandas,scipy",
                          machine=True))
        )