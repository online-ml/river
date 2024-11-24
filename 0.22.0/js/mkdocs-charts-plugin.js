// Adapted from https://github.com/koaning/justcharts/blob/main/justcharts.js
async function fetchSchema(url){
    var resp = await fetch(url);
    var schema = await resp.json();
    return schema
}
 
function checkNested(obj /*, level1, level2, ... levelN*/) {
    var args = Array.prototype.slice.call(arguments, 1);
  
    for (var i = 0; i < args.length; i++) {
      if (!obj || !obj.hasOwnProperty(args[i])) {
        return false;
      }
      obj = obj[args[i]];
    }
    return true;
  }
  

function classnameInParents(el, classname) {
    // check if class name in any parents
    while (el.parentNode) {
        el = el.parentNode;
        if (el.classList === undefined) {
            continue;
        }
        if (el.classList.contains(classname) ){
            return true;
        }
    }
    return false;
}

function findElementInParents(el, classname) {
    while (el.parentNode) {
        el = el.parentNode;
        if (el.classList === undefined) {
            continue;
        }
        if (el.classList.contains(classname) ){
            return el;
        }
    }
    return null;
}

function findProperChartWidth(el) {

    // mkdocs-material theme uses 'md-content'
    var parent = findElementInParents(el, "md-content")
    
    // mkdocs theme uses 'col-md-9'
    if (parent === undefined || parent == null) {
        var parent = findElementInParents(el, "col-md-9")        
    }
    if (parent === undefined || parent == null) {
        // we can't find a suitable content parent
        // 800 width is a good default
        return '800'
    } else {
        // Use full width of parent
        // Should bparent.offsetWidth - parseFloat(computedStyle.paddingLeft) - parseFloat(computedStyle.paddingRight) e equilavent to width: 100%
        computedStyle = getComputedStyle(parent)
        return parent.offsetWidth - parseFloat(computedStyle.paddingLeft) - parseFloat(computedStyle.paddingRight) 
    }
}

function updateURL(url) {

    // Strip anchor from URL if present
    let anchorIndex = url.indexOf('#');
    if (anchorIndex !== -1) {
        url = url.substring(0, anchorIndex);
    }

    // detect if absolute URL:
    // credits https://stackoverflow.com/a/19709846
    var r = new RegExp('^(?:[a-z]+:)?//', 'i');
    if (r.test(url)) {
        return url;
    }

    // If 'use_data_path' is set to true
    // schema and data urls are relative to
    // 'data_path', not the to current page
    // We need to update the specified URL
    // to point to the actual location relative to current page
    // Example:
    // Actual location data file: docs/assets/data.csv
    // Page: docs/folder/page.md
    // data url in page's schema: assets/data.csv
    // data_path in plugin settings: ""
    // use_data_path in plugin settings: True
    // path_to_homepage: ".." (this was detected in plugin on_post_page() event)
    // output url: "../assets/data.csv"
    if (mkdocs_chart_plugin['use_data_path'] == "True")  {
        let new_url = window.location.href

        // Strip anchor from URL if present
        let anchorIndex = new_url.indexOf('#');
        if (anchorIndex !== -1) {
            new_url = new_url.substring(0, anchorIndex);
        }
        
        new_url = new_url.endsWith('/') ? new_url.slice(0, -1) : new_url;
        
        if (mkdocs_chart_plugin['path_to_homepage'] != "") {
            new_url += "/" + mkdocs_chart_plugin['path_to_homepage']
        }

        new_url = new_url.endsWith('/') ? new_url.slice(0, -1) : new_url;
        new_url += "/" + url
        new_url = new_url.endsWith('/') ? new_url.slice(0, -1) : new_url;

        if (mkdocs_chart_plugin['data_path'] != "") {
            new_url += "/" + mkdocs_chart_plugin['data_path']
        }

        return new_url
    }
    return url;
}

const bodyelement = document.querySelector('body');

function getTheme() {
    // Get theme according to mkdocs-material's color scheme
    const materialColorScheme = bodyelement.getAttribute('data-md-color-scheme');
    if (materialColorScheme) {
        return mkdocs_chart_plugin['integrations']['mkdocs_material']['themes_dark'].includes(materialColorScheme)
            ? mkdocs_chart_plugin['vega_theme_dark']
            : mkdocs_chart_plugin['vega_theme'];
    }
    // Get theme according to user's preferred color scheme on the browser or OS
    if (window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
            ? mkdocs_chart_plugin['vega_theme_dark']
            : mkdocs_chart_plugin['vega_theme'];
    }
    // Fall back to light theme
    return mkdocs_chart_plugin['vega_theme'];
}

var vegalite_charts = [];

function embedChart(block, schema) {

    // Make sure the schema is specified
    let baseSchema = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    }
    schema = Object.assign({}, baseSchema, schema);

    // If width is not set at all, 
    // default is set to 'container'
    // Note we inserted <vegachart style='width: 100%'>..
    // So 'container' will use 100% width
    if (!('width' in schema)) {
        schema.width = mkdocs_chart_plugin['vega_width']
    }

    // Set default height if not specified
    // if (!('height' in schema)) {
    //     schema.height = mkdocs_chart_plugin['default_height']
    // }

    // charts widths are screwed in content tabs (thinks its zero width)
    // https://squidfunk.github.io/mkdocs-material/reference/content-tabs/?h=
    // we need to set an explicit, absolute width in those cases
    // detect if chart is in tabbed-content:
    if (classnameInParents(block, "tabbed-content")) {
      var chart_width = schema.width || 'notset';
      if (isNaN(chart_width)) {
          schema.width = findProperChartWidth(block);
      }
    }

    // Update URL if 'use_data_path' is configured
    if (schema?.data?.url !== undefined) {
        schema.data.url = updateURL(schema.data.url)
    }
    if (schema?.spec?.data?.url !== undefined) {
        schema.spec.data.url = updateURL(schema.spec.data.url)
    }
    // see docs/assets/data/geo_choropleth.json for example
    if (schema.transform) {
        for (const t of schema.transform) { 
            if (t?.from?.data?.url !== undefined) {
                t.from.data.url = updateURL(t.from.data.url)
            }
        }
    }
    

    // Save the block and schema
    // This way we can re-render the block
    // in a different theme
    vegalite_charts.push({'block' : block, 'schema': schema});

    // Render the chart
    vegaEmbed(block, schema, {
        actions: false, 
        "theme": getTheme(),
        "renderer": mkdocs_chart_plugin['vega_renderer']
    });
}

// Adapted from 
// https://facelessuser.github.io/pymdown-extensions/extensions/superfences/#uml-diagram-example
// https://github.com/koaning/justcharts/blob/main/justcharts.js
const chartplugin = className => {

    // Find all of our vegalite sources and render them.
    const blocks = document.querySelectorAll('vegachart');

    for (let i = 0; i < blocks.length; i++) {

      const block = blocks[i]
      const block_json = JSON.parse(block.textContent);

      // get the vegalite JSON
      if ('schema-url' in block_json) {

        var url = updateURL(block_json['schema-url'])
        fetchSchema(url).then(
            schema => embedChart(block, schema)
        );
      } else {
        embedChart(block, block_json);
      }

    }
  }

function updateCharts() {
    const theme = getTheme();
    for (let i = 0; i < vegalite_charts.length; i++) {
        vegaEmbed(vegalite_charts[i].block, vegalite_charts[i].schema, {
            actions: false,
            theme,
            "renderer": mkdocs_chart_plugin['vega_renderer']
        });
    }
}

// mkdocs-material has a dark mode including a toggle
// We should watch when dark mode changes and update charts accordingly

var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === "attributes") {
        
        if (mutation.attributeName == "data-md-color-scheme") {
            updateCharts();
        }

      }
    });
  });
observer.observe(bodyelement, {
attributes: true //configure it to listen to attribute changes
});

// Watch for user's preferred color scheme changes and update charts accordingly
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        updateCharts();
    });
}

// Load when DOM ready
if (typeof document$ !== "undefined") {
    // compatibility with mkdocs-material's instant loading feature 
    document$.subscribe(function() {
        chartplugin("vegalite")
    })
} else {
    document.addEventListener("DOMContentLoaded", () => {chartplugin("vegalite")})
}
