#!/usr/bin/env python2
from pyxflow.All import xf_All

A = xf_All("/Users/jdahm/xflow/demo/naca_0.xfa")
UG = A.GetPrimalState()
U = UG.GetVector()
M = A.Mesh
M.Plot()

