"""File with Top-level interfaces to XFlow"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API
#  2013-09-29 @dalle   : Added a plot method
#  2013-12-15 @dalle   : First version of xf_Plot class

# ------- Modules required -------
# Matplotlit essentials
import matplotlib.pyplot as plt
# Numeric functions
import numpy as np
# Line collection
from matplotlib.collections import LineCollection


# The background pyxflow workhorse module
from . import _pyxflow as px
# Mesh
from pyxflow.Mesh import xf_Mesh
# Geom
from pyxflow.Geom import xf_Geom
# DataSet
from pyxflow.DataSet import xf_DataSet, xf_VectorGroup, xf_Vector
# Plotting
from pyxflow.Plot import xf_Plot


class xf_Param:
    pass


class xf_EqnSet:

    def __init__(self, ptr):
        self._ptr = ptr


class xf_All:

    def __init__(self, fname, DefaultFlag=True):
        """
        All = xf_All(fname)

        """

        # Create an xf_All instance in memory
        self._ptr = px.ReadAllBinary(fname, DefaultFlag)

        # Get pointers to all members
        (Mesh_ptr, Geom_ptr, DataSet_ptr, Param_ptr,
            EqnSet_ptr) = px.GetAllMembers(self._ptr)

        # Shadow the members inside this class
        self.Mesh = xf_Mesh(ptr=Mesh_ptr)
        self.Geom = xf_Geom(ptr=Geom_ptr)
        self.EqnSet = xf_EqnSet(EqnSet_ptr)
        self.DataSet = xf_DataSet(ptr=DataSet_ptr)

    def __del__(self):
        px.DestroyAll(self._ptr)

    def GetPrimalState(self, TimeIndex=0):
        ptr = px.GetPrimalState(self._ptr, TimeIndex)
        return xf_VectorGroup(ptr)

    def Plot(self, **kwargs):
        """
        All = xf_All(...)
        h_t = All.Plot(xyrange=None, vgroup='State', scalar=None, mesh=False)

        INPUTS:
           xyrange : list of coordinates to plot (Note 1)
           vgroup  : title of vector group to use
           scalar  : name of scalar to plot (Note 2)
           mesh    : flag to draw a mesh

        OUTPUTS:
           h_t     : <matplotlib.pyplot.tripcolor> instance


        This is the plotting method for the xf_All class.  More capabilities
        will be added.

        NOTES:
           (1) The 'xyrange' keyword is specified in the form

                   [xmin, xmax, ymin, ymax]

               However, inputs such as `xyrange=(0,0,None,None)` are also
               acceptable.  In this case, the minimum value for both
               coordinates will be zero, but no maximum value will be
               specified.  Furthermore, alternate keys 'xmin', 'xmax', etc.
               override the values specified in 'range'.

           (2) The 'scalar' keyword may be any state that's in the cell
               interior list of states.  If the equation set is Navier-Stokes,
               the options Mach number, entropy, and pressure are also
               available.
        """
        # Versions:
        #  2013-09-29 @dalle   : First version

        # Create a plot object.
        h_p = xf_Plot(All=self, **kwargs)

        # Output the handles.
        return h_p


