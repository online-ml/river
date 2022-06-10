import json

import dominate
from dominate.tags import *
from slugify import slugify
from watermark import watermark

from river import datasets

with open("results.json") as f:
    benchmarks = json.load(f)

with open("details.json") as f:
    models = json.load(f)

with open("../docs/benchmarks/index.md", "w") as f:
    print_ = lambda x: print(x, file=f, end="\n\n")
    print_(
        """---
hide:
- navigation
---
"""
    )
    print_("# Benchmarks")

    print_("## Environment")
    print_(
        pre(watermark(python=True, packages="river,numpy,scikit-learn,pandas,scipy", machine=True))
    )

    imports = div()
    imports.add(
        link(
            href="https://unpkg.com/tabulator-tables@5.2.6/dist/css/tabulator.min.css",
            rel="stylesheet",
        )
    )
    imports.add(
        script(
            type="text/javascript",
            src="https://unpkg.com/tabulator-tables@5.2.6/dist/js/tabulator.min.js",
        )
    )
    print_(imports)

    print_(
        script(
            dominate.util.raw(
                """
        let baseColumns
        let metrics
        let columns
        """
            )
        )
    )

    for track_name, results in benchmarks.items():
        print_(f"## {track_name}")

        print_("### Results")
        print_(div(id=f"{slugify(track_name)}-results"))

        print_("### Datasets")
        for name, desc in models[track_name]["Dataset"].items():
            _details = details()
            _details.add(summary(name))
            _details.add(pre(desc))
            print_(_details)

        print_("### Models")
        for name, desc in models[track_name]["Model"].items():
            _details = details()
            _details.add(summary(name))
            _details.add(pre(desc))
            print_(_details)

        print_(
            script(
                dominate.util.raw(
                    f"""
    var results = {results}

    baseColumns = [
        "Dataset",
        "Model",
        "Memory",
        "Time"
    ]
    metrics = Object.keys(results[0]).filter(x => !baseColumns.includes(x)).sort();
    columns = [...baseColumns, ...metrics].map(x => ({{title: x, field: x}}))

    function formatBytes(bytes, decimals = 2) {{
        if (bytes === 0) return '0 Bytes'

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }}

    function msToTime(s) {{
        var ms = s % 1000;
        s = (s - ms) / 1000;
        var secs = s % 60;
        s = (s - secs) / 60;
        var mins = s % 60;
        var hrs = (s - mins) / 60;

        return (
            (hrs > 0 ? hrs + 'h ' : '') +
            (mins > 0 ? mins + 'm ' : '') +
            (secs > 0 ? secs + 's' : '') +
            (ms > 0 ? ' ' + Math.round(ms) + 'ms' : '')
        )
    }}

    columns.map((x, i) => {{
        if (x.title === 'Dataset') {{
            columns[i]["headerFilter"] = true
        }}
        if (x.title === 'Model') {{
            columns[i]["headerFilter"] = true
        }}
        if (x.title === 'Memory') {{
            columns[i]["formatter"] = function(cell, formatterParams, onRendered){{
                return formatBytes(cell.getValue())
            }}
        }}
        if (x.title === 'Time') {{
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {{
                return msToTime(cell.getValue())
            }}
        }}
        if (['Accuracy', 'F1', 'MacroF1', 'MicroF1'].includes(x.title)) {{
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {{
                return (100 * cell.getValue()).toFixed(2) + "%"
            }}
        }}
        if (['MAE', 'RMSE', 'R2'].includes(x.title)) {{
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {{
                return cell.getValue().toFixed(3)
            }}
        }}
    }})

    new Tabulator('#{slugify(track_name)}-results', {{
        data: results,
        layout: 'fitColumns',
        columns: columns
    }})
    """
                )
            )
        )
