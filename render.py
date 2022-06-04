import json
import dominate
from dominate.tags import *

with open('results.json') as f:
    benchmarks = json.load(f)

_html = html()
_html.add(link(href="https://unpkg.com/tabulator-tables@5.2.6/dist/css/tabulator.min.css", rel="stylesheet"))
_html.add(script(type="text/javascript", src="https://unpkg.com/tabulator-tables@5.2.6/dist/js/tabulator.min.js"))

_body = _html.add(body())

for track_name, results in benchmarks.items():
    _body.add(h2(track_name))
    _body.add(div(id=f"results"))
    _body.add(script(dominate.util.raw(f"""
    var results = {results}

    var table = new Tabulator('#results', {{
        data: results,
        layout: 'fitColumns',
        columns: Object.keys(results[0]).map(x => ({{title: x, field: x}}))
    }})
    """)))

print(_body)

with open('benchmarks.html', 'w') as f:
    print(_html, file=f)
