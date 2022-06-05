import json
import dominate
from dominate.tags import *
from river import datasets

with open('results.json') as f:
    benchmarks = json.load(f)

_html = html()
_html.add(link(href="https://unpkg.com/tabulator-tables@5.2.6/dist/css/tabulator.min.css", rel="stylesheet"))
_html.add(script(type="text/javascript", src="https://unpkg.com/tabulator-tables@5.2.6/dist/js/tabulator.min.js"))

_body = _html.add(body())
_body.add(h1("Online machine learning benchmarks"))

for track_name, results in benchmarks.items():
    _body.add(h2(track_name))
    _body.add(h3("Datasets"))
    # for dataset_name in sorted(set(r["Dataset"] for r in results)):
    #     dataset = eval(f"datasets.{dataset_name}()")
    #     detail = _body.add(details())
    #     detail.add(summary(dataset_name))
    #     detail.add(pre(repr(dataset)))
    _body.add(h3("Results"))
    _body.add(div(id=f"results"))
    _body.add(script(dominate.util.raw(f"""
    var results = {results}

    let baseColumns = [
        "Dataset",
        "Model",
        "Memory",
        "Time"
    ]
    let metrics = Object.keys(results[0]).filter(x => !baseColumns.includes(x)).sort();
    let columns = [...baseColumns, ...metrics].map(x => ({{title: x, field: x}}))

    function formatBytes(bytes, decimals = 2) {{
        if (bytes === 0) return '0 Bytes'

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }}

    function msToTime(s) {{
        function pad(n, z) {{
            z = z || 2;
            return ('00' + n).slice(-z);
        }}

        var ms = s % 1000;
        s = (s - ms) / 1000;
        var secs = s % 60;
        s = (s - secs) / 60;
        var mins = s % 60;
        var hrs = (s - mins) / 60;

        return pad(hrs) + ':' + pad(mins) + ':' + pad(secs) + '.' + pad(ms, 3);
    }}

    columns.map((x, i) => {{
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
        if (['Accuracy', 'F1'].includes(x.title)) {{
            columns[i]["formatter"] = function(cell, formatterParams, onRendered) {{
                return (100 * cell.getValue()).toFixed(2) + "%"
            }}
        }}
    }})

    var table = new Tabulator('#results', {{
        data: results,
        layout: 'fitColumns',
        columns: columns
    }})
    """)))

print(_html)

with open('benchmarks.html', 'w') as f:
    print(_html, file=f)
