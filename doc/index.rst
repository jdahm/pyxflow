.. pyXFlow documentation master file, created by
   sphinx-quickstart on Wed Jan 29 20:08:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyXFlow Documentation
=====================

Welcome to pyXFlow, which provides a Python module called :mod:`pyxflow` for
interfacing with XFlow.  In addition to providing an API of sorts for XFlow,
:mod:`pyxflow` contains a number of other utilities.  The main focus of these
is several utilities for creating graphics.  Now it's possible to make vector
graphics easily from XFlow results!


Contents
========

.. toctree::
    :maxdepth: 2
    :numbered:

    All
    Mesh
    Geom
    DataSet
    Plot

Installation
============

To install the :mod:`pyxflow` package, you will first need to meet several
prerequisites.  First, `XFlow <http://xflow.engin.umich.edu/>`_ must be
installed and compiled.  Second, you will need to have
`Git <http://git-scm.com/>`_ installed.  Ton istall Git on a Debian-based
system such as Ubuntu, simply run the following command in a terminal.

    .. code-block:: console
        
        $ sudo apt-get install git

Finally, there are two Python packages that are required.  The first of these
is the Python development backage, which is usually called `python-dev`.
The second package is `matplotlib <http://matplotlib.org/>`_, which is usually
a package of the same name.

An additional package that can make using :mod:`pyxflow` more intuitive is
called `IPython <http://ipython.org/>`_.  The main reason that IPython is
recommended is that it allows you to press the Tab key to autocomplete the
names of various :mod:`pyxflow` commands.

Once your system is prepared by following the above steps, download the
source code by running the following command.

    .. code-block:: console
        
        $ git clone https://github.com/jdahm/pyxflow.git

This will create a folder called `pyxflow` as a subdirectory of your current
working directory.  Before compiling, a configuration script must be run.  In
most cases, simply run the following.

    .. code-block:: console

        $ cd pyxflow
        $ ./configure -xflow path/to/xflow
        $ make

If the path to XFlow is `/home/user/xflow` (replacing `user` with your
username) or if it matches the value of `$HOME/xflow`, the commands can be
further simplified.

    .. code-block:: console

        $ cd pyxflow
        $ ./configure
        $ make






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

