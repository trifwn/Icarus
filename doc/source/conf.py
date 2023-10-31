# Configuration file for the Sphinx documentation builder.
#
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

# from ICARUS import __version__

##############################################################################
# General information about the project.
##############################################################################
project = "ICARUS"
copyright = "2023, Tryfonas Themas"
author = "Tryfonas Themas"
release = "Latest"

##############################################################################
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
##############################################################################

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    'autoclasstoc',
    'sphinx.ext.mathjax',
    "sphinx.ext.doctest",
    'sphinx.ext.napoleon',
    "sphinx.ext.todo",
    'sphinx_sitemap',
    'sphinx_design',
    "sphinx_markdown_parser",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_nb",
]

# Do some matplotlib config in case users have a matplotlibrc that will break
# things
matplotlib.use('agg')
plt.ioff()

# The suffix(es) of source filenames.
source_suffix: dict[str, str] = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# Templates
templates_path: list[str] = ["_templates"]

# The master toctree document.
master_doc = "index"

exclude_patterns: list[Any] = []

##############################################################################
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
##############################################################################

html_css_files: list[str] = [
    "css/custom.css",
]
html_static_path: list[str] = ["_static"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/trifwn/ICARUS",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_additional_pages: dict[str, Any] = {}
html_extra_path: list[str] = ['robots.txt']
html_use_modindex = True
html_domain_indices = False
html_copy_source = False
html_file_suffix = '.html'

##############################################################################
# -- Options for LaTeX output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output
##############################################################################

latex_elements: dict[str, str] = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

latex_documents = [
    ("index", "ICARUS.tex", "ICARUS Documentation", "Tryfonas Themas", "manual"),
]

##############################################################################
# Nitpicky mode - Warn about all broken links.
##############################################################################

nitpicky = True

##############################################################################
# Autodoc settings
##############################################################################

autodoc_default_options = {
    "inherited-members": None,
}

add_module_names = True

# Docstring settings
numpydoc_show_class_members = False
numpydoc_use_plots = True
# noindex option for numpydoc
numpydoc_class_members_toctree = False

##############################################################################
# Autosummary settings
##############################################################################

autosummary_generate = True

##############################################################################
# Notebook tutorials with MyST-NB
##############################################################################

nb_execution_mode = "auto"


##############################################################################
# PATHS
##############################################################################

# Specify the location of your code
import os
import sys

sys.path.insert(0, os.path.abspath("/home/tryfonas/data/Uni/Software/hermes/ICARUS"))
# sys.path.insert(0, os.path.abspath("/home/tryfonas/data/Uni/Software/hermes/examples"))
# sys.path.insert(0, os.path.abspath("/home/tryfonas/data/Uni/Software/hermes/cli"))
