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

    def Plot(self, scalar=None, **kwargs):
        """
        Plot the mesh and scalar from available data.
        
        :Call:
            >>> plot = All.Plot(scalar=None, **kwargs)

        :Parameters:
            All: :class:`pyxflow.All.xf_All`
                XFlow all object Python representation
            scalar: str
                Name of scalar to plot
                A value of `None` uses the default scalar.
                A value of `False` prevents plotting of any scalar.

        :Returns:
            plot: :class:`pyxflow.Plot.xf_Plot`
                pyXFlow plot instance with mesh and scalar handles
        
        :Kwargs:
            mesh: bool
                Whether or not to plot the mesh
            plot: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            role: str
                Identifier for the vector to use for plot
                The default value is `'ElemState'`.
            order: int
                Interpolation order for mesh faces
            vgroup: :class:`pyxflow.DataSet.xf_VectorGroup`
                Vector group to use for plot
                A value of `None` results in using the primal state.
                The behavior of this keyword argument is subject to change.
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        :See also:
            :func:`pyxflow.DataSet.xf_Vector.Plot()`
            :func:`pyxflow.Mesh.xf_Mesh.Plot()`
        """
        # Versions:
        #  2013-09-29 @dalle   : First version
        #  2014-02-09 @dalle   : Using xf_Vector.Plot() and xf_Mesh.Plot()
        
        # Extract the plot handle.
        plot = kwargs.get("plot")
        # Process the plot handle.
        if plot is None:
            # Initialize a plot.
            kwargs["plot"] = xf_Plot()
        elif not isinstance(plot, xf_Plot):
            raise IOError("Plot handle must be instance of " +
                "pyxflow.plot.xf_Plot")
        # Determine the vector group to use.
        UG = kwargs.get("vgroup")
        if UG is None:
            # Use the default vector group (PrimalState).
            UG = self.GetPrimalState()
        # Plot the mesh.
        if kwargs.get("mesh", True) is True:
            kwargs["plot"] = self.Mesh.Plot(**kwargs)
        # Plot the scalar.
        if scalar is not False and UG is not None:
            kwargs["scalar"] = scalar
            kwargs["plot"] = UG.Plot(self.Mesh, self.EqnSet, **kwargs)
        # Return the plot handle.
        return kwargs["plot"]


