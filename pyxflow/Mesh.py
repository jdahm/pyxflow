"""
The *Mesh* module contains the primary interface for XFlow's *xf_Mesh* structs,
which, not surprisingly, describe the meshes used for XFlow solutions.

Included in this module are several methods to access internal *xf_Mesh*
functions.  In other words, it provides an API for XFlow mesh methods.  In
addition, this module enables interactive or scripted mesh deformation by
shifting the node locations and rewriting the mesh to file.
"""

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
from pyxflow.Plot import xf_Plot, GetXLims

# ------- CLASSES -------
# --- Class to represent the (full) mesh ---


class xf_Mesh:
    """
    A Python class for XFlow mesh objects.
    
    If an instance is created with a pointer and no file name, the mesh is
    processed from the existing *xf_Mesh* struct (through the XFlow API).
    
    If both the file name and the pointer are ``None``, an empty *xf_Mesh* is
    created.
    
    :Call:
        >>> Mesh = xf_Mesh(fname=None, ptr=None)
    
    :Parameters:
        *fname*: :class:`str`
            Name of mesh file to read (usually ends in *.gri*)
        *ptr*: :class:`int`
            Pointer to existing *xf_Mesh* struct
    
    :Returns:
        *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
            Instance of the *xf_Mesh* interface
    
    :Data members:
        *_ptr*: :class:`int`
            Pointer to *xf_Mesh* struct being interfaced
        *Dim*: :class:`int`, ``1``, ``2``, or ``3``
            Dimension of the current mesh
        *nNode*: :class:`int`
            Number of nodes in the mesh
        *Coord*: :class:`numpy.array`, (*nNode*, *Dim*)
            Array of coordinates for each node
        *nBFaceGroup*: :class:`int`
            Number of boundary face groups
        *BFaceGroup*: :class:`pyxflow.Mesh.BFaceGroup` list
            List of boundary face group objects
        *nElemGroup*: :class:`int`
            Number of element groups in the mesh
        *ElemGroup*: :class:`pyxflow.Mesh.ElemGroup` list
            List of element groups in the mesh
    
    :Examples:
        To read a mesh file without also reading a solution, this class
        can be used with a *.gri* file.
        
            >>> Mesh = xf_Mesh("naca_quad.gri")
        
        The other way that meshes are instantiated usually involves reading
        the mesh component of an *.xfa* file.
        
            >>> All = xf_All("naca_adapt.xfa")
            >>> Mesh = xf_Mesh(All._Mesh)
            
        In fact, this is exactly how ``All.Mesh`` is created, and ``Mesh``
        in this case will be identical to ``All.Mesh``.  Furthermore,
        changes to one will affect the other.
    """

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
        Mesh initialization method
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
        
        Elements that do not have at least one node within the plot window
        are not plotted.  See :func:`pyxflow.Plot.GetXLims` for a thorough
        description of how the plot window can be created.
        
        :Call:
            >>> plot = Mesh.Plot(plot=None, **kwargs)
        
        :Parameters:
            *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh to be plotted
            *plot*: :class:`pyxflow.Plot.xf_Plot`
                Overall mesh handle to plot
                
        :Returns:
            *plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            *order*: :class:`int`
                Interpolation order for mesh faces
            *line_options*: :class:`dict`
                Options for :class:`matplotlib.pyplot.LineCollection`
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        
        """
        
        # Process the plot handle.
        if plot is None:
            # Initialize a plot.
            plot = xf_Plot()
        elif not isinstance(plot, xf_Plot):
            raise IOError("Plot handle must be instance of " +
                "pyxflow.plot.xf_Plot")
        # Use specified defaults for plot window if they exist.
        kwargs.setdefault('xmindef', plot.xmin)
        kwargs.setdefault('xmaxdef', plot.xmax)
        # Get the limits based on the Mesh and keyword args
        xLimMin, xLimMax = GetXLims(self, **kwargs)
        # Save the plot limits.
        plot.xmin = xLimMin
        plot.xmax = xLimMax
        
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
        # Return the plot handle.
        return plot
        


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
