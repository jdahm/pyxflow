# Packages
from distutils.core import setup, Extension
import ConfigParser
import json

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read("../config.cfg")

# The most important parameter: path to XFlow
xflow_home = config.get("xflow", "home")

# Compiler and linker options
cflagstrs = config.get("compiler", "cflags")
cflags = [str(x) for x in cflagstrs.split(' ')]

ldflagstrs = config.get("compiler", "rdynamic")
ldflags = [str(x) for x in ldflagstrs.split(' ')]

libstrs = config.get("xflow", "extra_libs")
extra_libs = [str(x) for x in libstrs.split(' ')]

includestrs = config.get("compiler", "include_dirs")
include_dirs = [str(x) for x in includestrs.split(' ')]

# Add the appropriate XFlow library to the list
libs = ["xfSerial"]

# Add xflow to the include_dirs (avoids having empty strings).
if include_dirs == ['']:
	include_dirs = [xflow_home+"/include"]
else:
	include_dirs = [xflow_home+"/include"] + include_dirs

# Assemble the information for the module
_pyxflow = Extension("_pyxflow",
    include_dirs = include_dirs,
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

# Compile and link
setup(
    name="python-xflow",
    version="1.0",
    description="This package is a python interface for xflow",
    ext_modules=[_pyxflow])
