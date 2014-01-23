#!/usr/bin/env python2

import os, ConfigParser
import subprocess as sp
import shutil, glob

config = ConfigParser.SafeConfigParser()
config.read("../config.cfg")

pythonexec = config.get("python", "exec")

print "Building XFlow equation sets for python..."
# Clean-up the existing build directory
shutil.rmtree("build", ignore_errors=True)
sp.call([pythonexec, "build.py"], cwd=os.getcwd()+"/lib")

print "Executing setup..."
sp.call([pythonexec, "setup.py", "build"])

print "Moving the module into place..."
dirs = glob.glob("build/lib*")
if len(dirs) > 1:
    raise ValueError("More than one build directory found")
libdir = dirs[0]
lib = libdir+"/_pyxflow.so"
shutil.move(lib, "./_pyxflow.so")

print "Removing the build directory..."
shutil.rmtree("build")
