#!/usr/bin/env python2

import subprocess as sp
import ConfigParser
import os, re

config = ConfigParser.SafeConfigParser()
config.read("../../config.cfg")

with open("Makefile.in", mode="w") as f:
    f.write("XF_HOME = {}".format(config.get("xflow", "home")))

sp.call(["make"])
