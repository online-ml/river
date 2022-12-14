import json
import pandas as pd


def render_df(df:pd.DataFrame, measure:str)-> dict:
    unique_datasets = list(df['dataset'].unique())
    res = {
        "data" : {"values": df.to_dict(orient="records")},
        "width": "container",
        "height": 500,
        "mark": "line",
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
        "encoding": {
            "x": {
                "name": "Instance",
                "field": "step",
                "type": "quantitative"
            },
            "y": {
                "field": measure,
                "type": "quantitative"
            },
            "color": {
                "field": "model",
                "scale": {"scheme": "category20b"}
            },
            "opacity": {
                "condition": {"param": "models", "value": 1},
                "value": 0.2
            }
        }
    }
    return res


if __name__ == '__main__':

    with open("../docs/benchmarks/index.md", "w", encoding='utf-8') as f:
        print_ = lambda x: print(x, file=f, end="\n\n")
        print_("# Benchmarks")

        print_("## Binary Classification")
        print_("### F1 Scores")
        print_("```vegalite")
        print_(json.dumps(render_df(pd.read_csv("Binary classification.csv"),"F1")))
        print_("```")

        print_("### Accuracy Scores")
        print_("```vegalite")
        print_(json.dumps(render_df(pd.read_csv("Binary classification.csv"),"Accuracy")))
        print_("```")

        print_("## Multi-Class Classification")