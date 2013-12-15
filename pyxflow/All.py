"""File with Top-level interfaces to XFlow"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API

# ------- Modules required -------
# Matplotlit essentials
import matplotlib.pyplot as plt
# Numeric functions
import numpy as np
# Line collection
from matplotlib.collections import LineCollection

# The background pyxflow workhorse module
import _pyxflow as px
# Mesh
from pyxflow.Mesh import xf_Mesh
# Geom
from pyxflow.Geom import xf_Geom
# DataSet
from pyxflow.DataSet import xf_DataSet
# Plotting routines
# from pyxflow.Plot import xf_Plot




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
        h_p = xf_Plot(self, **kwargs)
        
        # Output the handles.
        return h_p
        


# Class for xf_Plot objects
class xf_Plot:
    
    # Class initialization file.
    def __init__(self, All=None, fname=None, **kwargs):
        """
        h_p = xf_Plot(All=None, fname=None, **kwargs)
        
        INPUTS:
           All   : xf_All object
           fname : path to a '.xfa' file
        
        OUTPUTS:
           h_p   : instance of xf_Plot class
           
        KEYWORD ARGUMENTS:
           (See xf_Plot.Plot)
        """
        
        # Load the xf_All object if not given directly.
        if All is None:
            All = xf_All(fname)
        
        # Initialize figure handle.
        self.figure = None
        
        # Produce the initial plot.
        self.Plot(All, **kwargs)
        
        
        
        
        
    def Plot(self, All, xyrange=None, vgroup='State', scalar=None, mesh=False,
        xmin=None, xmax=None, ymin=None, ymax=None):
        """
        h_p = xf_Plot.Plot(All, xyrange=None, vgroup='State', scalar=None, mesh=False)
        
        INPUTS:
           All     : xf_All object
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
        
        # Check for a DataSet
        if not (All.DataSet.nData >= 1):
            raise IndexError("No DataSet found.")
        # Check that we have a vector group.
        if not (All.DataSet.Data[0].Type == 'VectorGroup'):
            raise TypeError("DataSet is not a xf_VectorGroup instance.")
        # This is for 2D right now!
        if All.Mesh.Dim != 2:
            raise NotImplementedError("3D plotting is not implemented.")
        
        # Process the window for plotting.
        if xyrange is not None:
            # Don't override values directly specified.
            if xmin is None: xmin = xyrange[0]
            if xmax is None: xmax = xyrange[1]
            if ymin is None: ymin = xyrange[2]
            if ymax is None: ymax = xyrange[3]
        # Make sure the values are not None before handing to C function.
        if xmin is None: xmin = All.Mesh.Coord[:,0].min()
        if xmax is None: xmax = All.Mesh.Coord[:,0].max()
        if ymin is None: ymin = All.Mesh.Coord[:,1].min()
        if ymax is None: ymax = All.Mesh.Coord[:,1].max()
        
        # Get the titles of the vector groups available.
        UG_Titles = [D.Title for D in All.DataSet.Data]
        # Check for the requested vector group.
        if vgroup in UG_Titles:
            # Get the matching vector group.
            UG = All.DataSet.Data[UG_Titles.index(vgroup)].Data
        elif vgroup is None:
            # Get the first vector group.
            UG = All.DataSet.Data[0].Data
        else:
            # Error
            raise RuntimeError((
                "Unrecognized DataSet title '%s'" % vgroup))
        
        # Limits on plot window
        xlim = [xmin, xmax, ymin, ymax]
        # Get the calculated vector, triangulation, and mesh lines.
        X, u, T, L = px.PlotData(All._ptr, UG._ptr, xlim)
        # Convert mesh lines to NumPy array.
        L = np.asarray(L)
        
        # Pull the first vector.
        U = UG.Vector[0]
        # Get the scalar.
        M = U.get_scalar(u, scalar)
        
        # Draw the requested scalar.
        h_t = plt.tripcolor(X[:,0], X[:,1], T, M, shading='gouraud')
        
        # Check for a grid.
        if mesh is True:
            # Nx2 matrix of xy-coordinates for each element
            xx = (X[j,:] for j in L)
            # Make a collection of lines with the same properties.
            h_l = LineCollection(xx, linewidths=0.2, colors=(0,0,0,1))
            # Have to plot these manually
            plt.gca().add_collection(h_l)
        
        # return the handle
        self.state = h_t

