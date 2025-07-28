# Configuration file for the Sphinx documentation builder.
#
# For information on options, see
#   http://www.sphinx-doc.org/en/master/config
#

import os
import sys
import subprocess
import re
import datetime
import shutil
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('./ext'))

#import add_man_page_redirects
#import build_manpages_c2html
#import fix_man_page_edit_links
#import make_links_relative
#import update_htmlmap_links
#import fix_pydata_margins


if not os.path.isdir("images"):
    print("-----------------------------------------------------------------------------")
    print("ERROR")
    print("images directory does not seem to exist.")
    print("To clone the required repository, try")
    print("   make images")
    print("-----------------------------------------------------------------------------")
    raise Exception("Aborting because images missing")


# -- Project information -------------------------------------------------------

project = 'SuperLU'
copyright = '2003-%d, The Regents of the University of California, through Lawrence Berkeley National Laboratory' % datetime.date.today().year
author = 'The SuperLU Development Team'

# -- General configuration -----------------------------------------------------

# The information on the next line must also be the same in requirements.txt
needs_sphinx='5.3'
nitpicky = True  # checks internal links. For external links, use "make linkcheck"
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build*', 'images', 'Thumbs.db', '.DS_Store','community/meetings/pre-2023']
highlight_language = 'c'
numfig = True

# -- Extensions ----------------------------------------------------------------

extensions = [
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'sphinxcontrib.rsvgconverter',
    'myst_parser',
    'sphinx_remove_toctrees',
    'sphinx_design',
]

copybutton_prompt_text = '$ '

bibtex_bibfiles = ['others.bib', 'sparse.bib']

myst_enable_extensions = ["fieldlist", "dollarmath", "amsmath", "deflist", "colon_fence"]

remove_from_toctrees = ['manualpages/*/[A-Z]*','changes/2*','changes/3*']

# prevents incorrect WARNING: duplicate citation for key "xxxx" warnings
suppress_warnings = ['bibtex.duplicate_citation']

# -- Options for HTML output ---------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_logo_light = os.path.join('images', 'logos', 'superlu-logo.png')
# html_static_path = ['_static', html_logo_light]

# use much smaller font for h1, h2 etc. They are absurdly large in the standard style
# https://pydata-sphinx-theme.readthedocs.io/en/v0.12.0/user_guide/styling.html
html_css_files = [
    'css/custom.css',
]

html_logo = html_logo_light
# html_favicon = os.path.join('images', 'logos', 'superlu-logo')
#html_last_updated_fmt = r'%Y-%m-%dT%H:%M:%S%z (' + git_describe_version + ')'


# -- Options for LaTeX output --------------------------------------------------
latex_engine = 'xelatex'

# How to arrange the documents into LaTeX files, building only the manual.
latex_documents = [
        ('manual/index', 'manual.tex', False)
        ]

latex_additional_files = [
]

