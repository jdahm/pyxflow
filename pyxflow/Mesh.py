"""File to interface with XFlow mesh objects in various forms"""

# Versions:
#  2013-09-22 @dalle   : First version


# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
from . import _pyxflow as px
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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

    def Plot(self, **kwargs):
        if kwargs.get('xmin') is not None:
            xmin = kwargs['xmin']
        else:
            xmin = [None for i in range(self.Dim)]

        if kwargs.get('xmax') is not None:
            xmax = kwargs['xmax']
        else:
            xmax = [None for i in range(self.Dim)]

        for i in range(self.Dim):
            if xmin[i] is None:
                xmin[i] = self.Coord[:, i].min()
            if xmax[i] is None:
                xmax[i] = self.Coord[:, i].max()

        if kwargs.get('figure') is not None:
            self.figure = kwargs['figure']
        else:
            self.figure = plt.figure()

        if kwargs.get('axes') is not None:
            self.axes = kwargs['axes']
        else:
            self.axes = self.figure.gca()

        x, y, c = px.MeshPlotData(self._ptr, xmin, xmax)
        s = [0] * (len(c) - 1)
        for f in range(len(c) - 1):
            s[f] = zip(x[c[f]:c[f + 1]], y[c[f]:c[f + 1]])

        line_options = kwargs.get('line_options', {})

        c = LineCollection(s, colors=(0, 0, 0, 1), **line_options)
        self.axes.add_collection(c)

        if kwargs.get('reset_limits', True):
            self.axes.set_xlim(xmin[0], xmax[0])
            self.axes.set_ylim(xmin[1], xmax[1])

        self.figure.savefig("figure.pdf")

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
