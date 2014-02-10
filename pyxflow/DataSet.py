"""File to interface with XFlow xf_DataSet objects in various forms"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import pyxflow._pyxflow as px
# Matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Import plotting methods
import pyxflow.Plot

# ------- Class for xf_Geom objects -------


class xf_DataSet:

    """A Python class for XFlow xf_DataSet objects"""

    # Initialization methd
    def __init__(self, ptr=None, fname=None, Mesh=None):
        """
        DataSet = xf_DataSet(ptr=None, fname=None, Mesh=None)

        INPUTS:
           ptr     : integer pointer to existing C xf_DataSet struct
           fname   : file name for a '.data' file to read the xf_DataSet from
           Mesh    : pointer to xf_Mesh, required if reading from file

        OUTPUTS:
           DataSet : an instance of the xf_DataSet Python class

        This function initializes a DataSet object in one of three ways.  The
        main method is to use a pointer that was previously created.  If the
        `ptr` key is not `None`, the function assumes the DataSet already exists
        on the heap and will read from it.  If `ptr` is `None` and both `fname`
        and `Mesh` are not `None`, the function will attempt to read the DataSet
        from file.  Finally, in all other cases, an empty DataSet is allocated.
        """
        # Versions:
        #  2013-09-25 @dalle   : First version

        # Set the defaults.
        self.nData = 0

        # Check the parameters.
        if ptr is not None:
            # Set the pointer.
            self._ptr = ptr
            self.owner = False
        elif fname is not None and Mesh is not None:
            # Read the file and get the pointer.
            ptr = px.ReadDataSetFile(Mesh, fname)
            # Set it.
            self._ptr = ptr
            self.owner = True
        else:
            # Create an empty geom.
            ptr = px.CreateDataSet()
            # Set the pointer
            self._ptr = ptr
            self.owner = True
            # Exit the function
            return None
        
        # Get the number of components.
        self.nData = px.nDataSetData(self._ptr)
        # Get the components
        self.Data = [xf_Data(self._ptr, i) for i in range(self.nData)]

    # xf_DataSet destructor method
    def __del__(self):
        """
        xf_DataSet destructor

        This function reminds the pyxflow module to clean up the C
        xf_DataSet object when the Python object is deleted.
        """
        # Version:
        #  2013-09-25 @dalle   : First version

        if self.owner and self._ptr is not None:
            px.DestroyDataSet(self._ptr)


# ---- Class for xf_Data struts ----
class xf_Data:

    """A Python class for XFlow xf_Data objects"""

    # Initialization method
    def __init__(self, DataSet, i=None):
        """
        Data = xf_Data(DataSet, i=None)

        INPUTS:
           DataSet : pointer to xf_DataSet struct
           i       : index of xf_Data to use

        OUTPUTS:
           Data    : an instance of the xf_Data class
               .Title : title of the xf_Data object
               .Type  : xf_Data type, see xfe_DataName
               .Data  : instance of xf_[Vector,VectorGroup] class
               ._Data : pointer to corresponding DataSet->Data
        """
        # Versions:
        #  2013-09-26 @dalle   : First version

        # Set the initial fields.
        self.Title = None
        self.Type = None
        self.Data = None
        self._ptr = None
        # Check for bad inputs.
        if DataSet is None:
            return None

        # Read from data if appropriate
        if i is not None:
            # Fields
            self.Title, self.Type, self._ptr, _Data = px.GetData(DataSet, i)

        # Do a switch on the type
        if self.Type == 'VectorGroup':
            # Assign the vector group
            self.Data = xf_VectorGroup(_Data)


# ---- Class for xf_VectorGroup ----
class xf_VectorGroup:

    """A Python class for XFlow xf_VectorGroup objects"""

    # Initialization method
    def __init__(self, ptr):
        """
        VG = xf_VectorGroup(ptr)

        INPTUS:
           ptr : pointer to xf_VectorGroup (DataSet->D->Data)

        OUTPUTS:
           VG  : instance of xf_VectorGroup object

        This function creates a VectorGroup object from a pointer.
        """
        # Versions:
        #  2013-09-26 @dalle   : First version

        # Check the pointer.
        if ptr is None:
            raise NameError
        # Set the pointer.
        self._ptr = ptr
        # Get the pointers to the vectors.
        self.nVector, V = px.GetVectorGroup(ptr)
        # Get the vectors.
        #self.V = V
        self.Vector = [xf_Vector(Vi) for Vi in V]

    # Method to get a plot based on the name of the role.
    def GetVector(self, role="ElemState"):
        _ptr = px.GetVectorFromGroup(self._ptr, role)
        return xf_Vector(_ptr)
        
    # Method to plot (passes information to xf_Vector.Plot())
    def Plot(self, Mesh, EqnSet, role="ElemState", **kwargs):
        """
        Plot a scalar from a vector group.
        
        :Call:
            >>> plot = UG.Plot(Mesh, EqnSet, role="ElemState", **kwargs)
        
        :Parameters:
            UG: :class:`pyxflow.DataSet.xf_VectorGroup`
                Vector group containing vector to plot
            Mesh: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh for geometry data required for plotting
            EqnSet: :class:`pyxflow.EqnSet.xf_EqnSet`
                Equation set data
            role: str
                Identifier for the vector to use for plot
                
        :Returns:
            plot: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            plot: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            scalar: str
                Name of scalar to plot
            order: int
                Interpolation order for mesh faces
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        :See also:
            :func:`pyxflow.DataSet.xf_Vector.Plot()`
        """
        # Versions:
        #  2014-02-09 @dalle   : First version
        
        # Get the vector.
        U = self.GetVector(role)
        # Plot based on the plot.
        plot = U.Plot(Mesh, EqnSet, **kwargs)
        # Return the plot handle.
        return plot


# ---- Class for xf_Vector ----
class xf_Vector:

    """A Python class for XFlow xf_Vector objects"""

    # Initialization method
    def __init__(self, ptr):
        """
        V = xf_Vector(ptr)

        INPUTS:
           ptr : ponter to xf_Vector

        OUTPUTS:
           V   : instance of xf_Vector class

        This function creates an xf_Vector object from pointer.
        """
        # Versions:
        #  2013-09-26 @dalle   : First version

        # Check the pointer.
        if ptr is None:
            raise NameError
        # Set the pointer.
        self._ptr = ptr
        # Get the information and pointers to GenArrays.
        (self.nArray, self.Order, self.Basis,
         self.StateName, GA) = px.GetVector(ptr)
        # Get the GenArrays
        self.GenArray = GA
        self.GenArray = [xf_GenArray(G) for G in GA]
    
    # Plotting method
    def Plot(self, Mesh, EqnSet, scalar=None, plot=None, **kwargs):
        """
        Plot a scalar.
        
        :Call:
            >>> plot = U.Plot(Mesh, EqnSet, scalar=None, plot=None, **kwargs)
            
        :Parameters:
            U: :class:`pyxflow.DataSet.xf_Vector`
                Vector containing scalar data to plot
            Mesh: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh for geometry data required for plotting
            EqnSet: :class:`pyxflow.EqnSet.xf_EqnSet`
                Equation set data
            scalar: str
                Name of scalar to plot
            plot: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
                
        :Returns:
            plot: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            order: int
                Interpolation order for mesh faces
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        :Notes:
            When `scalar` is `None`, the first scalar available in the vector
            will be used.  This is often `'Density'`.
        """
        # Mesh dimension
        dim = Mesh.Dim
        # Process the plot.
        if plot is None:
            # Initialize a plot.
            plot = pyxflow.Plot.xf_Plot()
        elif not isinstance(plot, pyxflow.Plot.xf_Plot):
            raise IOError("Plot handle must be instance of " +
                "pyxflow.plot.xf_Plot")
        # Use specified defaults for plot window if they exist.
        kwargs.setdefault('xmindef', plot.xmin)
        kwargs.setdefault('xmaxdef', plot.xmax)
        # Get the limits based on the Mesh and keyword args
        xmin, xmax = pyxflow.Plot.GetXLims(Mesh, **kwargs)
        # Save the plot limits.
        plot.xmin = xmin
        plot.xmax = xmax
            
        # Check for an existing mesh plot.
        if plot.scalar is not None:
            # Delete it!
            plot.scalar.remove()
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
        # Scalar name; break on default
        Name = scalar
        # Process the colormap option...
        colormap = kwargs.get('colormap', plt.cm.jet)
        # Get the mesh nodes and subnodes and their scalar values.
        x, y, tri, scalar = px.ScalarPlotData(
            self._ptr, Mesh._ptr, EqnSet._ptr, Name, xmin, xmax, Order)
        # Check the dimension.
        if dim > 1:
            # Create a set of triangles with gradient colors.
            T = Triangulation(x, y, triangles=tri)
            p = plot.axes.tripcolor(T, scalar, shading='gouraud', cmap=colormap)
            # Store the tripcolor handle.
            plot.scalar = p
        else:
            # Plot the value versus x.
            plot.scalar = plot.axes.plot(x, scalar)
        # Apply the bounding box that was created earlier.
        if kwargs.get('reset_limits', True):
            plot.axes.set_xlim(xmin[0], xmax[0])
            if dim > 1:
                plot.axes.set_ylim(xmin[1], xmax[1])
        # Draw if necessary.
        if plt.isinteractive():
            plt.draw()
        # Return the plot.
        return plot


# ---- Class for xf_GenArray ----
class xf_GenArray:

    """A Python class for XFlow xf_GenArray objects"""

    # Initialization method
    def __init__(self, ptr):
        """
        GA = xf_GenArray(ptr)

        INPUTS:
           ptr : pointer to xf_GenArray

        OUTPUTS:
           GA  : instance of xf_GenArray class

        This function creates an xf_GenArray instance, which holds the actual
        data of most xf_Vector structs.
        """
        # Versions:
        #  2013-09-26 @dalle   : First version

        # Check the pointer.
        if ptr is None:
            raise NameError
        # Set the pointer.
        self._ptr = ptr
        # Get the GenArray info from the API.
        GA = px.GetGenArray(ptr)
        # Assign the parameters.
        # Number of elements in element group
        self.n = GA["n"]
        # Number of (fixed) degrees of freedom per element
        self.r = GA["r"]
        # Number of (variable) degrees of freedom per element
        self.vr = GA["vr"]
        # Integer data
        self.iValue = GA["iValue"]
        # Real data
        self.rValue = GA["rValue"]
