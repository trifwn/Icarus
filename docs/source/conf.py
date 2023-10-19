# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pty import master_open

project = 'ICARUS'
copyright = '2023, Tryfonas Themas'
author = 'Tryfonas Themas'
release = 'Latest'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Enable the autodoc extension
extensions = [
    'sphinx_markdown_parser',
    'myst_parser',
    'sphinx.ext.autodoc',
]

# Specify the location of your code (add this to the end of the file)
import os
import sys
sys.path.insert(0, os.path.abspath('/home/tryfonas/data/Uni/Software/hermes/ICARUS'))


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_css_files = [
    'css/custom.css',
]
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']


# -- Options for LaTeX output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    ('index', 'ICARUS.tex', 'ICARUS Documentation',
     'Tryfonas Themas', 'manual'),
]

# Autodoc settings
autodoc_member_order = 'bysource'
# autodoc_member_order = 'groupwise'
autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'show-inheritance']

# def skip(app, what, name, obj, skip, options):
#     if name == "__init__":
#         return False
#     return skip

# def setup(app):
#     app.connect("autodoc-skip-member", skip)

