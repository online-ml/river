import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dominate.tags import pre
from watermark import watermark
import textwrap


# ---------- Paths & helpers ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_BENCH_DIR = REPO_ROOT / "docs" / "benchmarks"


def slugify(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------- Plot rendering ----------
def _infer_measures(df: pd.DataFrame) -> List[str]:
    """Infer metric columns from a benchmark dataframe.

    Prefer numeric columns excluding the common identifier columns.
    Fallback to columns[4:] if needed to remain backward-compatible.
    """
    id_cols = {"dataset", "model", "step"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in {"step"}]
    measures = [c for c in numeric_cols if c not in id_cols]
    if not measures:
        # fallback to the previous behavior
        measures = list(df.columns)[4:]
    return [c for c in measures if c in df.columns]


def _palette(models: List[str]) -> Dict[str, str]:
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    return {m: colors[i % len(colors)] for i, m in enumerate(models)}


def render_df_blocks(df_path: Path, id_prefix: str | None = None) -> List[tuple[str, str]]:
    """Render the benchmark CSV into responsive Plotly HTML blocks per dataset.

    - Uses Markdown for dataset headings outside this function.
    - No Plotly figure title to avoid duplicate headings.
    Returns: list of (dataset_name, html_block)
    """
    df = pd.read_csv(str(df_path))

    required = {"dataset", "model", "step"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        warn_html = f"<div class='admonition warning'>Missing required columns: {missing}</div>"
        return [("", warn_html)]

    # Stable ordering
    unique_datasets = sorted(df["dataset"].unique())
    unique_models = sorted(df["model"].unique())
    measures = _infer_measures(df)

    palette = _palette(unique_models)
    blocks: List[tuple[str, str]] = []

    for dataset in unique_datasets:
        dataset_df = df[df["dataset"] == dataset]
        if dataset_df.empty:
            continue

        nrows = max(1, len(measures))
        fig = make_subplots(
            rows=nrows,
            cols=1,
            subplot_titles=[m.replace("_", " ").title() for m in measures] if measures else None,
            shared_xaxes=True,
            vertical_spacing=0.06,
        )

        for model in unique_models:
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
                            line=dict(color=palette[model], width=2),
                            legendgroup=str(model),
                            showlegend=(i == 0),  # one legend entry per model
                            hovertemplate=(
                                f"Model: {model}<br>Step: %{{x}}<br>{measure.replace('_',' ').title()}: %{{y:.4f}}<extra></extra>"
                            ),
                        ),
                        row=(i + 1),
                        col=1,
                    )

        # Layout
        fig.update_layout(
            height=max(420, 180 * nrows),
            showlegend=bool(measures),
            # No Plotly title; the page shows a Markdown heading instead
            template="plotly_white",
            margin=dict(l=56, r=36, t=36, b=48),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Axes and empty-state annotation
        if measures:
            fig.update_xaxes(title_text="Instance", row=nrows, col=1)
            for i, measure in enumerate(measures):
                fig.update_yaxes(title_text=measure.replace("_", " ").title(), row=i + 1, col=1)
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
        prefix = f"{slugify(id_prefix)}-" if id_prefix else ""
        fig_div_id = f"plot-{prefix}{dataset_slug}"
        config = {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d", "autoScale2d"],
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

        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config=config,
            div_id=fig_div_id,
            validate=False,
        )

        blocks.append(
            (
                dataset,
                f"""
<div class=\"benchmark-plot\" id=\"{fig_div_id}-container\">\n  {html}\n</div>\n""",
            )
        )

    return blocks


def _page_header() -> str:
        return """---
"""


def _render_track(track_name: str, track_details: Dict, csv_dir: Path) -> str:
    out: List[str] = []
    out.append(f"## {track_name}")

    csv_name = track_name.replace(" ", "_").lower() + ".csv"
    df_path = find_first_existing([Path.cwd() / csv_name, csv_dir / csv_name])

    # If CSV is in CWD, move it to docs so MkDocs can serve it from a stable place
    if df_path and df_path.parent == Path.cwd():
        target = csv_dir / csv_name
        print(f"Moving {df_path} -> {target}")
        safe_move(df_path, target)
        df_path = target

    if df_path and df_path.exists():
        df_md = (
            pd.read_csv(str(df_path))
            .groupby(["model", "dataset"])
            .last()
            .drop(columns=["track", "step"])
            .reset_index()
            .rename(columns={"model": "Model", "dataset": "Dataset"})
            .to_markdown(index=False)
        )

        out.append("")
        out.append(f"""=== "Table" """)
        out.append("")
        out.append(f"""{textwrap.indent(df_md, '    ')}""")
        out.append("")
        out.append(f"""=== "Chart" """)


        for dataset_name, html_block in render_df_blocks(df_path, id_prefix=slugify(track_name)):
            if dataset_name:
                # Insert explicit blank lines to ensure Markdown parsing after HTML blocks
                out.append("")
                out.append(f"### {dataset_name}")
                out.append("")
            out.append(html_block)
    else:
        out.append(f"<div class='admonition note'>CSV {csv_name} not found. Skipping visualization.</div>")

    # Collapsible metadata
    out.append("### Datasets")
    for dataset_name, dataset_details in track_details.get("Dataset", {}).items():
        out.append("<details class=\"bench-details\">")
        out.append(f"<summary class=\"bench-summary\">{dataset_name}</summary>")
        out.append(str(pre(dataset_details, _class="bench-pre dataset-pre")))
        out.append("</details>")

    out.append("### Models")
    for model_name, model_details in track_details.get("Model", {}).items():
        out.append("<details class=\"bench-details\">")
        out.append(f"<summary class=\"bench-summary\">{model_name}</summary>")
        out.append(str(pre(model_details, _class="bench-pre model-pre")))
        out.append("</details>")
    return "\n\n".join(out)


def main() -> int:
    print("Starting benchmarks render...")
    print(f"CWD: {os.getcwd()}")

    ensure_dir(DOCS_BENCH_DIR)

    # Locate and ensure details.json is in docs/benchmarks
    details_src = find_first_existing([Path.cwd() / "details.json", DOCS_BENCH_DIR / "details.json"])
    if details_src is None:
        print("ERROR: details.json not found in current directory or docs/benchmarks/")
        return 1
    if details_src.parent != DOCS_BENCH_DIR:
        print("Moving details.json into docs/benchmarks/")
        safe_move(details_src, DOCS_BENCH_DIR / "details.json")

    details_path = DOCS_BENCH_DIR / "details.json"
    try:
        details: Dict = json.loads(details_path.read_text())
    except Exception as e:
        print(f"ERROR: Failed to load details.json: {e}")
        return 1

    index_md_path = DOCS_BENCH_DIR / "index.md"
    with index_md_path.open("w", encoding="utf-8") as f:
        def print_(x: str) -> None:
            print(x, file=f, end="\n\n")

        print_(_page_header())

        # Tracks
        for track_name, track_details in details.items():
            print_(_render_track(track_name, track_details, DOCS_BENCH_DIR))

        # Environment
        print_("# Environment")
        print_(
            str(
                pre(
                    watermark(
                        python=True, packages="river,numpy,scikit-learn,pandas,scipy,plotly", machine=True
                    )
                )
            )
        )

    print(f"Wrote {index_md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
