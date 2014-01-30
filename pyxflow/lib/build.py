#!/usr/bin/env python2

import subprocess as sp
import ConfigParser
import os, re
import glob

# Load the configuration file.
config = ConfigParser.SafeConfigParser()
config.read("../../config.cfg")

# Write the "Makefile.in" from scratch.
with open("Makefile.in", mode="w") as f:
    f.write("XF_HOME = %s\n" % config.get("xflow", "home"))
    f.write("CFLAGS = %s\n" % config.get("compiler", "cflags"))
    f.write("LDFLAGS = %s\n" % config.get("compiler", "ldflags"))

# Run the make command directly.
sp.call(["make"])
