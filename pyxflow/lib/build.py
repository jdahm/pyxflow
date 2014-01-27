#!/usr/bin/env python2

import subprocess as sp
import ConfigParser
import os, re
import glob

# Load the configuration file.
config = ConfigParser.SafeConfigParser()
config.read("../../config.cfg")

# Find the directory with the object files.
dirs = glob.glob("../build/temp*")
# Check the number of directories.
if len(dirs) > 1:
    raise ValueError("More than one build directory found.")
# The directory that contains the objects.
objdir = dirs[0]
# Find the objects present there.
fnames = glob.glob(objdir + "/*.o")
# Turn this list of files into a string.
objstr = " ".join(fnames)
print "\n\n\n\nfiles: "
print objstr
print "\n\n\n\n"

# Write the "Makefile.in" from scratch.
with open("Makefile.in", mode="w") as f:
    f.write("RDYNAMIC = %s\n" % config.get("compiler", "rdynamic"))
    f.write("XF_HOME = %s\n" % config.get("xflow", "home"))
    f.write("OBJ_LIST = %s" % objstr)

# Run the make command directly.
sp.call(["make"])
