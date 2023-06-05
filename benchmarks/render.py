import json
import shutil
import textwrap
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
            "values": df.to_dict(orient="records")
            # "url": f"benchmarks/{df_path.name}"
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
    with open("details.json") as f:
        details = json.load(f)

    for track_name, track_details in details.items():
        track_dir = Path(f"../docs/benchmarks/{track_name}")
        track_dir.mkdir(exist_ok=True)
        with open(f"../docs/benchmarks/{track_name}/index.md", "w", encoding="utf-8") as f:
            print_ = lambda x: print(x, file=f, end="\n\n")

            print_(f"# {track_name}")

            # Move the dataset from the benchmarks folder to the docs folder
            csv_name = track_name.replace(" ", "_").lower()
            shutil.copy(f"{csv_name}.csv", f"../docs/benchmarks/{track_name}/{csv_name}.csv")

            df_path = Path(f"../docs/benchmarks/{track_name}/{csv_name}.csv")

            df_md = (
                pd.read_csv(str(df_path))
                .groupby(["model", "dataset"])
                .last()
                .drop(columns=["track", "step"])
                .reset_index()
                .rename(columns={"model": "Model", "dataset": "Dataset"})
                .to_markdown(index=False)
            )

            print_(
                f"""

=== "Table"

{textwrap.indent(df_md, '    ')}

=== "Chart"

    *Try reloading the page if something is buggy*

    ```vegalite
{textwrap.indent(json.dumps(render_df(df_path), indent=2), '    ')}
    ```

            """
            )

            print_("## Datasets")
            for dataset_name, dataset_details in track_details["Dataset"].items():
                print_(f'???- abstract "{dataset_name}"')
                print_(textwrap.indent(dataset_details, "    "))
                print_(f"<span />")
            print_("## Models")
            for model_name, model_details in track_details["Model"].items():
                print_(f'???- example "{model_name}"')
                print_(
                    f"    <pre>{textwrap.indent(model_details, '    ').replace('    ', '', 1)}</pre>"
                )
                print_(f"<span />")

            print_("## Environment")
            print_(
                pre(
                    watermark(
                        python=True, packages="river,numpy,scikit-learn,pandas,scipy", machine=True
                    )
                )
            )
