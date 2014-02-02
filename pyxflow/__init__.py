"""
pyXFlow
=======

pyXFlow is a python interface to some of the functionality of libXF, the
high-order adaptive finite-element library.
"""

__version__ = 0.1

from .All import xf_All
from .Mesh import xf_Mesh
from .Geom import xf_Geom
from .DataSet import xf_DataSet
from .Plot import xf_Plot, set_colormap
