
# No shebang because build.py determines which python executable to use.

# Packagess
from distutils.core import setup, Extension
import ConfigParser
import json

# Get a get/set type object.
config = ConfigParser.SafeConfigParser()
# Read the configuration options.
config.read("../PX_CONFIG")

# The most important parameter: path to XFlow
xflow_home = config.get("xflow", "home")
# Equation set to link...
# This is static.
eqnset = config.get("xflow", "eqnset")
# Additional libraries
libstrs = config.get("xflow", "libs")
cflagstrs = config.get("compiler", "cflags")
cflags = [str(x) for x in json.loads(cflagstrs)]
ldflagstrs = config.get("compiler", "ldflags")
ldflags = [str(x) for x in json.loads(ldflagstrs)]
extra_libs = [str(x) for x in json.loads(libstrs)]
# Add the appropriate XFlow library to the list.
libs = ["xfSerial"] + extra_libs


# Assemble the information for the module
_pyxflow = Extension("_pyxflow",
    include_dirs = [xflow_home+"/include"],
    libraries = libs,
    library_dirs = [xflow_home+"/lib"],
    runtime_library_dirs = [xflow_home+"/lib"],
    extra_compile_args = cflags,
    extra_link_args = ldflags,
    extra_objects = [xflow_home+"/build/src/xf_EqnSetHook.o"],
    sources = [
        "_pyxflowmodule.c",
        "px_Geom.c",
        "px_Mesh.c",
        "px_DataSet.c",
        "px_Plot.c",
        "px_All.c"])

# Compile and link.
setup(
    name = "python-xflow",
    version = "1.0",
    description = "This package is a python interface for xflow",
    ext_modules = [_pyxflow])
