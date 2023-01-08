import json
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from dominate.tags import pre
from watermark import watermark


def render_df(df_path: Path) -> dict:
    df = pd.read_csv(str(df_path))

    unique_datasets = list(df["dataset"].unique())
    measures = list(df.columns)[4:]
    res = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {
            # "values": df.to_dict(orient="records")
            "url": f"benchmarks/{df_path.name}"
        },
        "params": [
            {"name": "models", "select": {"type": "point", "fields": ["model"]}, "bind": "legend"},
            {
                "name": "Dataset",
                "value": unique_datasets[0],
                "bind": {"input": "select", "options": unique_datasets},
            },
            {"name": "grid", "select": "interval", "bind": "scales"},
        ],
        "transform": [{"filter": {"field": "dataset", "equal": {"expr": "Dataset"}}}],
        "repeat": {"row": measures},
        "spec": {
            "width": "container",
            # "height": "container",
            "mark": "line",
            "encoding": {
                "x": {
                    "field": "step",
                    "type": "quantitative",
                    "axis": {
                        "titleFontSize": 18,
                        "labelFontSize": 18,
                        "title": "Instance",
                    },
                },
                "y": {
                    "field": {"repeat": "row"},
                    "type": "quantitative",
                    "axis": {"titleFontSize": 18, "labelFontSize": 18},
                },
                "color": {
                    "field": "model",
                    "type": "ordinal",
                    "scale": {"scheme": "category20b"},
                    "title": "Models",
                    "legend": {
                        "titleFontSize": 18,
                        "labelFontSize": 18,
                        "labelLimit": 500,
                    },
                },
                "opacity": {"condition": {"param": "models", "value": 1}, "value": 0.2},
            },
        },
    }
    return res


if __name__ == "__main__":

    if Path("details.json").exists():
        if Path("../docs/benchmarks/details.json").exists():
            Path("../docs/benchmarks/details.json").unlink()
        shutil.move("details.json", "../docs/benchmarks/details.json")
    details = json.load(open("../docs/benchmarks/details.json"))

    with open("../docs/benchmarks/index.md", "w", encoding="utf-8") as f:
        print_ = lambda x: print(x, file=f, end="\n\n")
        print_(
            """---
hide:
- navigation
---
"""
        )

        print_("# Benchmark")

        for track_name, track_details in details.items():
            print_(f"## {track_name}")
            csv_name = track_name.replace(" ", "_").lower()
            if Path(f"{csv_name}.csv").exists():
                if Path(f"../docs/benchmarks/{csv_name}.csv").exists():
                    Path(f"../docs/benchmarks/{csv_name}.csv").unlink()
                shutil.move(
                    f"{csv_name}.csv",
                    "../docs/benchmarks/",
                )

            df_path = Path(f"../docs/benchmarks/{csv_name}.csv")
            print_("```vegalite")
            print_(json.dumps(render_df(df_path), indent=2))
            print_("```")

            print_("### Datasets")
            for dataset_name, dataset_details in track_details["Dataset"].items():
                print_(f"<details>")
                print_(f"<summary>{dataset_name}</summary>")
                print_(pre(dataset_details))
                print_(f"</details>")
            print_("### Models")
            for model_name, model_details in track_details["Model"].items():
                print_(f"<details>")
                print_(f"<summary>{model_name}</summary>")
                print_(pre(model_details))
                print_(f"</details>")

        print_("# Environment")
        print_(
            pre(
                watermark(
                    python=True, packages="river,numpy,scikit-learn,pandas,scipy", machine=True
                )
            )
        )
