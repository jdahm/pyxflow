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

#h = M.Plot(xmin=[-0.1, -0.5], xmax=[1.1, 0.5])
h = A.Plot(scalar="Pressure", xlim=[-0.1,1.1], ylim=[-0.5,0.5])
#h = UG.Plot(M, E, role="ElemState", scalar="XMomentum", plot=h)
plt.show()
#f.savefig("figure.pdf", bbox_inches='tight')
