# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import inspect
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import inverse_thomson_scattering

project = 'TSADAR'
copyright = '2023, Avi Milder, Archis Joglekar'
author = 'Avi Milder, Archis Joglekar'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
#New theme is pydata
html_theme = 'pydata_sphinx_theme'
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_github_style",
    "sphinx_design",
    "sphinx.ext.intersphinx",
]
# options for sphinx_github_style
top_level = "inverse_thomson_scattering"
linkcode_blob = "head"
linkcode_url = r"https://github.com/ergodicio/inverse-thomson-scattering/"
linkcode_link_text = "Source"


def linkcode_resolve(domain, info):
    """Returns a link to the source code on GitHub, with appropriate lines highlighted"""

    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # for jitted stuff, get the original src
    if hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__

    # get the link to HEAD
    cmd = "git log -n1 --pretty=%H"
    try:
        # get most recent commit hash
        head = subprocess.check_output(cmd.split()).strip().decode("utf-8")

        # if head is a tag, use tag as reference
        cmd = "git describe --exact-match --tags " + head
        try:
            tag = subprocess.check_output(cmd.split(" ")).strip().decode("utf-8")
            blob = tag

        except subprocess.CalledProcessError:
            blob = head

    except subprocess.CalledProcessError:
        print("Failed to get head")  # so no head?
        blob = "main"

    linkcode_url = r"https://github.com/ergodicio/inverse-thomson-scattering/"
    linkcode_url = linkcode_url.strip("/") + f"/blob/{blob}/"
    linkcode_url += "{filepath}#L{linestart}-L{linestop}"

    # get a Path object representing the working directory of the repository.
    try:
        cmd = "git rev-parse --show-toplevel"
        repo_dir = Path(subprocess.check_output(cmd.split(" ")).strip().decode("utf-8"))

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Unable to determine the repository directory") from e

    # For ReadTheDocs, repo is cloned to /path/to/<repo_dir>/checkouts/<version>/
    if repo_dir.parent.stem == "checkouts":
        repo_dir = repo_dir.parent.parent

    # path to source file
    try:
        filepath = os.path.relpath(inspect.getsourcefile(obj), repo_dir)
        if filepath is None:
            return
    except Exception:
        return None

    # lines in source file
    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    # Fix links with "../../../" or "..\\..\\..\\"
    filepath = "/".join(filepath[filepath.find(top_level) :].split("\\"))

    final_link = linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop
    )
    print(f"Final Link for {fullname}: {final_link}")
    return final_link


# numpydoc_class_members_toctree = False
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__call__",
    "exclude-members": "__init__",
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }
# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
# New theme is pydata
html_theme = 'pydata_sphinx_theme'
#html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    #    'canonical_url': '',
    #    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    #"logo_link": "index.html",
    #TSADAR logo and tittle, both of which should redirect you to the homepage 
    "logo":{
        "text":"Home",
        "image_light": "_elfolder/lafoto.JPG",
        "image_dark": "_elfolder/lafoto.JPG",
    },
    #github icon in the nav bar that directs you to the repo
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ergodicio/inverse-thomson-scattering.git",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "header_links_before_dropdown": 4,
    #"logo_only": True,
    #remove secondary sidebar
    "secondary_sidebar_items":[],
    #"display_version": True,
    "show_nav_level": 2,
    "prev_next_buttons_location": "both",
    "style_external_links": False,
    "style_nav_header_background": "#3c4142",
    #"logo" : "TSADAR",
    #back to top button
    "back_to_top_button": True,
    # maintaining the search button in the nav bar
    "navbar_persistent" : "search-button-field",
    #location of the navigation bar
    "navbar_align": "left",
    # top left of the navigation bar will include logo and version 
    "navbar_start": ["navbar-logo"],
    #placing the search field at the top right of the navigation bar
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    #show the home page in the nav bar
    "home_page_in_toc": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
    }


#this will be used to make the light mode the default mode
html_context = {
    "default_mode": "auto"
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# the top of the navigation bar will say TASADAR
# the logo for TSADAR can be found inside a folder called "_elfolder" 
html_tittle = "TSADAR",
#html_logo = '_elfolder/lafoto.JPG'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "_static/images/desc_icon.ico"


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": ["globaltoc.html", "sidebar-nav-bs.html"]
}
#html_sidebars = {
#   "**": ["sbt-sidebar-nav.html", "sidebar-primary.html"]
#    "**": ["sbt-sidebar-nav.html"]
#}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "tsadar"
