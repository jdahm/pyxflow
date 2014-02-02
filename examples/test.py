#!/usr/bin/env python2
import os.path as op
import ConfigParser
from pyxflow.All import xf_All
import matplotlib.pyplot as plt

config = ConfigParser.SafeConfigParser()
config.read("../config.cfg")

A = xf_All(op.join(config.get("xflow", "home"), "demo/naca_Adapt_0.xfa"))
M = A.Mesh
E = A.EqnSet

UG = A.GetPrimalState()
U = UG.GetVector("ElemState")

#f = M.Plot(xmin=[-0.1, -0.5], xmax=[1.1, 0.5])
h = A.Plot(xlim=[-0.1,1.1], ylim=[-0.5,0.5])
#f = U.Plot(M, E, figure=f, xmin=[-0.1, -0.5], xmax=[1.1, 0.5], scalar="XMomentum")
plt.show()
#f.savefig("figure.pdf", bbox_inches='tight')
