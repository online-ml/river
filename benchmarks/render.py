from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
from dominate.tags import pre
from slugify.slugify import slugify
from watermark import watermark

# Professional color palette for models (Material Design inspired)
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def render_df(dataset_df: pd.DataFrame, measures: List[str], models: List[str],
              dataset: str, include_plotlyjs: bool = True) -> str:
    nrows = max(1, len(measures))
    fig = make_subplots(
        rows=nrows,
        cols=1,
        subplot_titles=[m.replace("_", " ").title() for m in
                        measures] if measures else None,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # Assign colors to models based on their index
    model_colors = {model: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, model in
                    enumerate(models)}

    for model in models:
        model_df = dataset_df[dataset_df["model"] == model]
        if model_df.empty:
            continue
        # Ensure sorted by step for nice lines
        model_df = model_df.sort_values("step")

        if measures:
            for i, measure in enumerate(measures):
                y = model_df[measure]
                if y.isna().all():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=model_df["step"],
                        y=y,
                        name=str(model),
                        mode="lines",
                        legendgroup=str(model),
                        showlegend=(i == 0),  # one legend entry per model
                        line=dict(color=model_colors[model], width=2.5),
                        hovertemplate=(
                            f"Model: {model}<br>Step: %{{x}}<br>{measure.replace('_', ' ').title()}: %{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=(i + 1),
                    col=1,
                )

    # Layout
    # Increase bottom margin to accommodate legend below the chart
    bottom_margin = 150 if measures else 48
    fig.update_layout(
        height=max(500, 180 * nrows + 150),
        showlegend=bool(measures),
        # No Plotly title; the page shows a Markdown heading instead
        template="plotly_white",
        margin=dict(l=56, r=36, t=36, b=bottom_margin),
        hovermode="x unified",
        plot_bgcolor="rgba(245, 245, 245, 0.5)",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
            size=12,
            color="#424242"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    # Axes and empty-state annotation
    if measures:
        fig.update_xaxes(title_text="Instance", row=nrows, col=1)
        for i, measure in enumerate(measures):
            fig.update_yaxes(title_text=measure.replace("_", " ").title(),
                             row=i + 1, col=1)
    else:
        fig.add_annotation(
            text="No numeric metrics found in CSV.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14),
        )

    # HTML conversion
    dataset_slug = slugify(dataset)
    fig_div_id = f"plot-{dataset_slug}"
    config = {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d",
                                   "autoScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{dataset_slug}_benchmark",
            "height": 600,
            "width": 1000,
            "scale": 2,
        },
        "scrollZoom": False,
        "staticPlot": False,
    }
    plotlyjs_setting = "require" if include_plotlyjs else False

    html = fig.to_html(
        include_plotlyjs=plotlyjs_setting,
        full_html=False,
        config=config,
        div_id=fig_div_id,
        validate=False,
    )

    return (
        f"""<div class=\"benchmark-plot\" id=\"{fig_div_id}-container\">\n  {html}\n</div>\n""")


if __name__ == "__main__":
    with open("details.json") as f:
        details = json.load(f)

    for track_name, track_details in details.items():
        track_dir = Path(f"../docs/benchmarks/{track_name}")
        track_dir.mkdir(exist_ok=True)
        with open(f"../docs/benchmarks/{track_name}/index.md", "w",
                  encoding="utf-8") as f:

            def print_(x):
                return print(x, file=f, end="\n\n")


            print_(f"# {track_name}")

            # Move the dataset from the benchmarks folder to the docs folder
            csv_name = track_name.replace(" ", "_").lower()
            shutil.copy(f"{csv_name}.csv",
                        f"../docs/benchmarks/{track_name}/{csv_name}.csv")

            df_path = Path(f"../docs/benchmarks/{track_name}/{csv_name}.csv")

            df = pd.read_csv(str(df_path))

            unique_datasets = list(df["dataset"].unique())
            measures = list(df.columns)[4:]
            unique_models = sorted(df["model"].unique())

            # Track whether we've already included Plotly.js
            first_plot = True

            for dataset in unique_datasets:
                dataset_df = df[df["dataset"] == dataset]
                print_(f"## {dataset}")

                print_(f"### Summary")
                df_md = (
                    dataset_df
                    .groupby(["model"])
                    .last()
                    .drop(columns=["track", "step", "dataset"])
                    .reset_index()
                    .rename(columns={"model": "Model"})
                    .to_markdown(index=False)
                )
                print_(df_md)
                print_(f"### Charts")
                print_(render_df(dataset_df=dataset_df, measures=measures,
                                 models=unique_models, dataset=dataset,
                                 include_plotlyjs=first_plot))
                first_plot = False

            print_("## Datasets")
            for dataset_name, dataset_details in track_details[
                "Dataset"].items():
                print_(f'???- abstract "{dataset_name}"')
                print_(textwrap.indent(dataset_details, "    "))
                print_("<span />")
            print_("## Models")
            for model_name, model_details in track_details["Model"].items():
                print_(f'???- example "{model_name}"')
                print_(
                    f"    <pre>{textwrap.indent(model_details, '    ').replace('    ', '', 1)}</pre>"
                )
                print_("<span />")

            print_("## Environment")
            print_(
                pre(
                    watermark(
                        python=True,
                        packages="river,numpy,scikit-learn,pandas,scipy",
                        machine=True
                    )
                )
            )
