#!/usr/bin/env python2
from pyxflow.All import xf_All
import matplotlib.pyplot as plt

A = xf_All("/Users/jdahm/xflow/demo/naca_0.xfa")
M = A.Mesh
E = A.EqnSet

UG = A.GetPrimalState()
U = UG.GetVector("ElemState")

f = M.Plot(xmin=[-0.1, -0.1], xmax=[0.1, 0.1])
f = U.Plot(M, E, figure=f, xmin=[-0.1, -0.1], xmax=[0.1, 0.1], scalar="XMomentum")
f.savefig("figure.pdf", bbox_inches='tight')
