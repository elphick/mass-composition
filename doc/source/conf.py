# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
from plotly.io._sg_scraper import plotly_sg_scraper
import plotly.io as pio

pio.renderers.default = 'sphinx_gallery'

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'mass-composition'
copyright = '2022, Greg Elphick'
author = 'Greg Elphick'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',  # to document the api
              'sphinx.ext.viewcode',  # to add view code links
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',  # for parsing numpy/google docstrings
              'sphinx_gallery.gen_gallery',  # to generate a gallery of examples
              'sphinx_autodoc_typehints',
              'myst_parser'  # for parsing md files
              ]

autosummary_generate = True
sphinx_gallery_conf = {
    'filename_pattern': r'\.py',
    'ignore_pattern': r'__init__\.py',
    'examples_dirs': '../../examples',  # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    # 'subsection_order': ExplicitOrder(['basic_usage',
    #                                    'math_operations',
    #                                    'plot_demo']),
    'within_subsection_order': FileNameSortKey,
    'image_scrapers': ('matplotlib', )
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_style = '_static/my_theme.css'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
