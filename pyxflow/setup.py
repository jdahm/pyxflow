# Packages
from distutils.core import setup, Extension
import ConfigParser
import json
import os.path as op

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read("../config.cfg")

# Path to this fule
directory=op.dirname(op.realpath(__file__))

# The most important parameter: path to XFlow
xflow_home = config.get("xflow", "home")
xflow_lib = op.join(xflow_home, "lib")
xflow_include = op.join(xflow_home, "include")

# Path to pyxflow equation sets
pyxflow_lib = op.join(directory, "lib")

# Compiler and linker options
cflagstrs = config.get("compiler", "extra_cflags")
cflags = [str(x) for x in cflagstrs.split(' ')]

ldflagstrs = config.get("compiler", "extra_ldflags")
ldflags = [str(x) for x in ldflagstrs.split(' ')]

includestrs = config.get("compiler", "extra_include_dirs")
include_dirs = [str(x) for x in includestrs.split(' ')]

libstrs = config.get("xflow", "extra_libs")
extra_libs = [str(x) for x in libstrs.split(' ')]

# Add the appropriate XFlow library to the list
libs = ["xfSerial"]

# Add xflow to the include_dirs
include_dirs.append(op.join(xflow_home, "include"))

# filter out empty strings from include_dirs
include_dirs = filter(None, include_dirs)

# Assemble the information for the module
_pyxflow = Extension("_pyxflow",
    include_dirs = filter(None, include_dirs),
    libraries = filter(None, libs),
    library_dirs = [xflow_lib],
    runtime_library_dirs = [xflow_lib, pyxflow_lib],
    extra_compile_args = filter(None, cflags),
    extra_link_args = filter(None, ldflags),
    extra_objects = [op.join(xflow_home, "build/src/xf_EqnSetHook.o")],
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
