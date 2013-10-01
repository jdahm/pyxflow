"""File with Top-level interfaces to XFlow"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API

# ------- Modules required -------
# Matplotlit essentials
import matplotlib.pyplot as plt
# The background pyxflow workhorse module
import _pyxflow as px
# Mesh
from pyxflow.Mesh import xf_Mesh
# Geom
from pyxflow.Geom import xf_Geom
# DataSet
from pyxflow.DataSet import xf_DataSet




class xf_Param:
    pass

class xf_EqnSet:

    def __init__(self, ptr):
        self._ptr = ptr


class xf_All:

    def __init__(self, fname, DefaultFlag=True):

        # Create an xf_All instance in memory
        self._ptr = px.ReadAllInputFile(fname, True)

        # Get pointers to all members
        (Mesh_ptr, Geom_ptr, DataSet_ptr, Param_ptr, 
            EqnSet_ptr) = px.GetAllMembers(self._ptr)

        # Shadow the members inside this class
        self.Mesh    = xf_Mesh(ptr=Mesh_ptr)
        self.Geom    = xf_Geom(ptr=Geom_ptr)
        self.EqnSet  = xf_EqnSet(EqnSet_ptr)
        self.DataSet = xf_DataSet(ptr=DataSet_ptr)
        

    def __del__(self):
        px.DestroyAll(self._ptr)
        
    def Plot(self, xyrange=None,
        xmin=None, xmax=None, ymin=None, ymax=None):
        """
        All = xf_All(...)
        h_t = All.Plot(xyrange=None)
        
        INPUTS:
           xyrange : list of coordinates to plot (Note 1)
        
        OUTPUTS:
           h_t     : <matplotlib.pyplot.tripcolor> instance
        
        
        This is the plotting method for the xf_All class.  More capabilities
        will be added.
        
        NOTES:
           (1) The 'xyrange' keyword is specified in the form
           
                   [xmin, xmax, ymin, ymax]
               
               However, inputs such as `xyrange=(0,0,None,None)` are also
               acceptable.  In this case, the minimum value for both coordinates
               will be zero, but no maximum value will be specified.
               Furthermore, alternate keys 'xmin', 'xmax', etc. override the
               values specified in 'range'.
           
        """
        # Versions:
        #  2013-09-29 @dalle   : First version
        
        # Check for a DataSet
        if not (self.DataSet.nData >= 1):
            raise IndexError("No DataSet found.")
        # Check that we have a vector group.
        if not (self.DataSet.Data[0].Type == 'VectorGroup'):
            raise TypeError("DataSet is not a xf_VectorGroup instance.")
        # This is for 2D right now!
        if self.Mesh.Dim != 2:
            raise NotImplementedError("3D plotting is not implemented.")
        
        # Process the window for plotting.
        if xyrange is not None:
            # Don't override values directly specified.
            if xmin is None: xmin = xyrange[0]
            if xmax is None: xmax = xyrange[1]
            if ymin is None: ymin = xyrange[2]
            if ymax is None: ymax = xyrange[3]
        # Make sure the values are not None before handing to C function.
        if xmin is None: xmin = self.Mesh.Coord[:,0].min()
        if xmax is None: xmax = self.Mesh.Coord[:,0].max()
        if ymin is None: ymin = self.Mesh.Coord[:,1].min()
        if ymax is None: ymax = self.Mesh.Coord[:,1].max()
        # Get the vector group.
        UG = self.DataSet.Data[0].Data
        # Limits on plot window
        xlim = [xmin, xmax, ymin, ymax]
        # Get the data and triangulation.
        X, u, T = px.InterpVector(self._ptr, UG._ptr, xlim)
        
        # Draw the plot
        # Using density for now.
        h_t = plt.tripcolor(X[:,0], X[:,1], T, u[:,0], shading='gouraud')
        
        # return the handle
        return h_t
        
        
        

