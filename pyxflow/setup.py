from distutils.core import setup, Extension
import ConfigParser
import json

config = ConfigParser.SafeConfigParser()
config.read("../config.cfg")

xflow_home = config.get("xflow", "home")
eqnset = config.get("xflow", "eqnset")
extra_libs = [str(x) for x in json.loads(config.get("xflow", "libs"))]
libs = ["xfSerial", eqnset] + extra_libs

_pyxflow = Extension("_pyxflow",
    include_dirs = [xflow_home+"/include"],
    libraries = libs,
    library_dirs = [xflow_home+"/lib"],
    runtime_library_dirs = [xflow_home+"/lib"],
    sources = ["_pyxflowmodule.c", "px_Geom.c", "px_Mesh.c", "px_All.c"])

setup(
    name = "python-xflow",
    version = "1.0",
    description = "This package is a python interface for xflow",
    ext_modules = [_pyxflow])
