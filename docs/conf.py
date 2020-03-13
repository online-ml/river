# Path setup

import os
import sys; sys.path.insert(0, os.path.abspath('..'))

# Project information

project = 'creme'
author = 'The creme developers'
copyright = '2020, the creme developers'
import creme
version = creme.__version__
from distutils.version import LooseVersion
release = LooseVersion(creme.__version__).vstring

# General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton'
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'friendly'
default_role = 'any'  # (used for this markup: `text`)

# Theming

import sphinx_material

extensions.append('sphinx_material')
html_theme = 'sphinx_material'
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_show_sourcelink = True
html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'localtoc.html', 'searchbox.html']
}
html_static_path = ['_static']
html_use_index = True
html_domain_indices = True
html_favicon = '_static/favicon.ico'
html_logo = '_static/creme_square.svg'

html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': f'creme v{version}',

    # Set you GA account ID to enable tracking
    'google_analytics_account': 'UA-63302552-3',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://creme-ml.github.io',

    # Set the color and the accent color
    #'color_primary': 'white',
    #'color_accent': 'red',

    'nav_links': [],

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/creme-ml/creme/',
    'repo_name': 'creme',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 1,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True
}

# nbsphinx

extensions.append('nbsphinx')
nbsphinx_execute = 'never'

# napolean

extensions.append('sphinx.ext.napoleon')
napoleon_use_rtype = False
napoleon_use_ivar = True

# autosummary

extensions.append('sphinx.ext.autosummary')
autoclass_content = 'class'
autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_default_options = {
    'show-inheritance': True
}

# intersphinx extension

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'surprise': ('https://surprise.readthedocs.io/en/stable/', None)
}

# MathJax

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': 'true'
    },
}
