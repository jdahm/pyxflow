pyXFlow
=======

This package provides python bindings to XFlow.


Dependencies
------------
Packages required for the core module pyxflow.*:
* XFlow (and its dependencies)

Packages required for the plotting library pyxflow.plot.*
* Numpy
* Matplotlib

TODO: figure out minimum versions required


Setup
-----
The steps to start using pyXFlow are:
* configure the code using the ./configure bash script.  If run without
    options, or with the "-help" flag, the options are printed
* compile the python C module that interfaces with XFlow by typing "make"
* build the documentation with "make doc"

Run configure script with ./configure with appropriate options.


Files/Directories
-----------------
Files:
* configure - configuration shell script, writes PX_CONFIG
* AUTHORS - a list of authors who have contributed to the project

Folders:
* pyxflow - top level of the pyxflow python module
* doc - sphinx documentation


Style
-----
Python uses 4 spaces by default, so that's what we should also use for the module extensions in C.

