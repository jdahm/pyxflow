"""File to interface with XFlow mesh objects in various forms"""

# Versions:
#  2013-09-22 @dalle   : First version
#  2014-02-07 @dalle   : Integrated @jdahm's plot methods


# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
from . import _pyxflow as px
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Import the plot class
import pyxflow.Plot

# ------- CLASSES -------
# --- Class to represent the (full) mesh ---


class xf_Mesh:

    """A Python class for XFlow mesh objects"""

    # Parameters
    _ptr = None
    Dim = 0
    nNode = 0
    Coord = None
    nIFace = 0
    IFace = None
    nBFaceGroup = 0
    BFaceGroup = None
    nElemGroup = 0
    ElemGroup = None

    # Method to initialize the object
    def __init__(self, fname=None, ptr=None):
        """
        Mesh = xf_Mesh(fname=None, ptr=None)

        INPUTS:
           fname : file name for a '.gri' file to read the mesh from
           ptr   : integer pointer to existing C xf_Mesh struct

        OUTPUTS:
           Mesh  : an instance of the xf_Mesh Python class

        This function initializes a Mesh object in one of three ways.  If
        the `gri` key is not `None`, the function will attempt to read a
        mesh from a text file.  If the `ptr` is not `None`, the function
        assumes the mesh already exists on the heap and will read from it.
        If both keys are `None`, an empty mesh will be created.  Finally,
        if both keys are not `None`, an exception is raised.
        """
        # Versions:
        #  2013-09-23 @dalle   : First version

        # Check the parameters.
        if fname is not None:
            if ptr is not None:
                raise NameError
            # Read the file and get the pointer.
            ptr = px.ReadGriFile(fname)
            # Set it.
            self._ptr = ptr
            self.owner = True
        elif ptr is not None:
            # Simply set the pointer.
            self._ptr = ptr
            self.owner = False
        else:
            # Create an empty mesh.
            ptr = px.CreateMesh()
            # Set the pointer.
            self._ptr = ptr
            self.owner = True
            # Exit the function with default properties.
            return None

        # Get the basic coordinate information.
        self.Dim, self.nNode, self.Coord = px.GetNodes(self._ptr)

        # Basic boundary condition info
        self.nBFaceGroup = px.nBFaceGroup(self._ptr)
        # Get the BFaceGroups
        self.BFaceGroup = [xf_BFaceGroup(ptr=self._ptr, i=i)
                           for i in range(self.nBFaceGroup)]

        # Basic element group info
        self.nElemGroup = px.nElemGroup(self._ptr)
        # Get the ElemGroups
        self.ElemGroup = [xf_ElemGroup(ptr=self._ptr, i=i)
                          for i in range(self.nElemGroup)]

    # Destructor method for xf_Mesh
    def __del__(self):
        """
        xf_Mesh destructor

        This function reminds the pyxflow module to clean up the C
        xf_Mesh object when the python object is deleted.
        """
        # Version:
        #  2013-09-23 @dalle   : First version

        if self.owner:
            px.DestroyMesh(self._ptr)

    # Plot method for mesh
    def Plot(self, plot=None, **kwargs):
        """Create a plot for an :class:`xf_Mesh` object.
        
        Elements that do not have at least one node with a coordinate between
        `xmin[i]` and `xmax[i]`, for `i` corresponding to each dimension in the
        mesh, are not plotted.
        
        :Call:
            >>> plot = Mesh.Plot(plot=None, **kwargs)
        
        :Parameters:
            Mesh: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh to be plotted
            plot: :class:`pyxflow.Plot.xf_Plot`
                Overall mesh handle to plot
                
        :Returns:
            plot : :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            order: int
                Interpolation order for mesh faces
            line_options: dict
                Options for matplotlib.pyplot.LineCollection
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        
        """
        
        # Get the limits based on the Mesh and keyword args
        xLimMin, xLimMax = pyxflow.Plot.GetXLims(self, **kwargs)
        # Process the plot.
        if plot is None:
            # Initialize a plot.
            plot = pyxflow.Plot.xf_Plot()
        elif not isinstance(plot, pyxflow.Plot.xf_Plot):
            raise IOError("Plot handle must be instance of " +
                "pyxflow.plot.xf_Plot")
        
        # Check for an existing mesh plot.
        if plot.mesh is not None:
            # Delete it!
            plot.mesh.remove()
        # Determine what figure to use.
        if kwargs.get('figure') is not None:
            # Use the input figure.
            plot.figure = kwargs['figure']
        elif plot.figure is None:
            # Follow norms of plotting programs; default is gcf().
            plot.figure = plt.gcf()
        # Determine what axes to use.
        if kwargs.get('axes') is not None:
            # Use the input value.
            plot.axes = kwargs['axes']
        else:
            # Normal plotting conventions for default
            plot.axes = plt.gca()
        # Plot order; apparently None leads to default below?
        Order = kwargs.get('order')

        # Get the plot data for each element.
        # Don't worry, eventually someone will explain what the hell c is.
        # It's a list of the node indices in each mesh element.
        x, y, c = px.MeshPlotData(self._ptr, xLimMin, xLimMax, Order)
        # Turn this into a list of coordinates.
        s = []
        for f in range(len(c) - 1):
            s.append(zip(x[c[f]:c[f+1]], y[c[f]:c[f+1]]))

        # Get any options that should be applied to the actual plot.
        line_options = kwargs.get('line_options', {})
        # Set the default color.
        line_options.setdefault('colors', (0,0,0,1))
        # Create the lines efficiently.
        hl = LineCollection(s, **line_options)
        # Plot them.
        plot.axes.add_collection(hl)
        # Apply the bounding box that was created earlier.
        if kwargs.get('reset_limits', True):
            plot.axes.set_xlim(xLimMin[0], xLimMax[0])
            plot.axes.set_ylim(xLimMin[1], xLimMax[1])
        # Store the handle to the line collection.
        plot.mesh = hl
        # Draw if necessary.
        if plt.isinteractive():
            plt.draw()
        


# --- Class for boundary face groups ---


class xf_BFaceGroup:

    """
    Boundary face group object for pyxflow, a Python interface for XFlow

    """
    # Versions:
    #   2013-09-24 @dalle   : _pyxflow version
    #   2013-09-29 @dalle   : changed ptr to Mesh instead of BFaceGroup

    # Initialization method
    def __init__(self, Title=None, nBFace=None, ptr=None, i=None):
        # Define the properties
        self.Title = Title
        self.nBFace = nBFace
        self.BFace = None
        self._ptr = None
        # Check for a pointer.
        if ptr is not None:
            # index
            if i is None:
                i = 0
            # Fields
            self.Title, self.nBFace, self._ptr = px.BFaceGroup(ptr, i)


# --- Class for boundary face groups ---
class xf_ElemGroup:

    """
    Element group object for pyxflow, a Python interface for XFlow

    """
    # Versions:
    #   2013-09-24 @dalle   : _pyxflow version
    #   2013-09-29 @dalle   : changed ptr to Mesh instead of ElemGroup

    # Initialization method
    def __init__(self, ptr=None, i=None):
        # Define the properties
        self.nElem = 0
        self.nNode = 0
        self.QOrder = 0
        self.QBasis = None
        self.Node = None
        # Check for a pointer.
        if ptr is not None:
            # index
            if i is None:
                i = 0
            # Fields
            (self.nElem, self.nNode, self.QOrder, self.QBasis,
                self.Node) = px.ElemGroup(ptr, i)


# --- Class for boundary faces ---
class xf_BFace:

    """
    Boundary face class for pyxflow, a Python interface for XFlow

    """
    # Initialization method

    def __init__(self, ElemGroup=0, Elem=0, Face=0, Orient=0):
        # Define the properties.
        self.ElemGroup = ElemGroup
        self.Elem = Elem
        self.Face = Face
        self.Orient = Orient
    # No methods for now
