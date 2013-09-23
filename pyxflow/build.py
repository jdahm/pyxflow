#!/usr/bin/env python2

import subprocess as sp
import ConfigParser
import shutil

config = ConfigParser.SafeConfigParser()
config.read("../config.cfg")

pythonexec = config.get("python", "exec")

print "Executing setup..."
sp.call([pythonexec, "setup.py", "build"])

print "Moving the module into place..."
libdir = "build/lib."+config.get("python", "arch")+"-"+config.get("python", "version")
lib = libdir+"/_pyxflow.so"
shutil.move(lib, "./_pyxflow.so")

print "Removing the build directory..."
shutil.rmtree("build")
