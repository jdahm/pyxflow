"""
File to interface with XFlow *xf_DataSet* objects in various forms

The *DataSet* module contains the primary interface for XFlow's *xf_DataSet*
structs, which contain information on XFlow solutions.  This includes the state,
which is the traditional type of CFD solution; the adjoint; and many other
things.

Two very important aspects of data sets are vector groups (*xf_VectorGroup*) and
vectors (*xf_Vector*).  Roughly, vectors contain the values of entities that
have values for each element in the solution (either interior elements or face
elements), and vector groups are, well, groups of vectors.  Although it is
possible to perform some tasks in pyXFlow without any understanding of data sets
and vectors (for example using the :func:`pyxflow.All.Plot` method and its
defaults), they are critical to a deeper investigation of XFlow solutions.

Included in this module are several methods to access internal *xf_DataSet*
functions.  In other words, it provides an API for XFlow data sets.
"""

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
    """
    A Python class for XFlow *xf_DataSet* objects
    
    The option to read from the pointer to an existing *xf_DataSet* is
    prioritized.  Alternatively, if both *fname* and *Mesh* are valid inputs,
    the data set is constructed by reading from a `.data.` file.
    
    :Call:
        >>> DS = xf_DataSet(ptr=None, fname=None, Mesh=None)
    
    :Parameters:
        *ptr*: :class:`int`
            Pointer to *xf_DataSet* from which to read
        *fname*: :class:`str`
            Name of `.data` file to read from
        *Mesh*: :class:`int` or :class:`pyxflow.Mesh.xf_Mesh`
            Pointer to *xf_Mesh* (``Mesh._ptr`` is used if it exists)
    
    :Data members:
        *DS._ptr*: :class:`int`
            Pointer to the XFlow *xf_DataSet*
        *DS.nData*: :class:`int`
            Number of XFlow *xf_Data* objects contained in the data set
        *DS.Data*: :class:`pyxflow.DataSet.xf_Data` list
            List of *xf_Data* interfaces
    """

    # Initialization methd
    def __init__(self, ptr=None, fname=None, Mesh=None):
        """
        Initialization method for *xf_DataSet*
        """
        # Versions:
        #  2013-09-25 @dalle   : First version

        # Set the defaults.
        self.nData = 0
        # Check if the Mesh was passed... rather than its pointer.
        if hasattr(Mesh, '_ptr'):
            Mesh = Mesh._ptr

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
        *xf_DataSet* destructor

        This function reminds the pyxflow module to clean up the C
        xf_DataSet object when the Python object is deleted.
        """
        # Version:
        #  2013-09-25 @dalle   : First version

        if self.owner and self._ptr is not None:
            px.DestroyDataSet(self._ptr)


# ---- Class for xf_Data struts ----
class xf_Data:
    """
    A Python class for XFlow *xf_Data* objects
    
    :Call:
        >>> D = xf_Data(DataSet, i)
    
    :Parameters:
        *DataSet*: :class:`int`
            Pointer to *xf_DataSet* struct
        *i*: :class:`int`
            Quasi-index of *xf_Data* struct to read; since *DataSet->Data* is
            actually a linked list, these are read starting with
            *DataSet->Data->Head*
    
    :Data members:
        *D._ptr*: :class:`int`
            Pointer to *xf_Data* struct
        *D.Title*: :class:`int`
            Name of the data object
        *D.Type*: :class:`str`
            Type of data contained, usually ``'VectorGroup'``
        *D.Data*: :class:`pyxflow.DataSet.xf_Vector` or :class:`pyxflow.DataSet.xf_VectorGroup`
            Instance of an object containing data
    
    :Examples:
        Data members are usually extracted from solutions (e.g., `.xfa` files),
        although reading from `.data` files is also possible.
        
            >>> All = xf_All('naca_adapt_0.xfa')
            >>> D = All.DataSet.Data[0]
            >>> D.Title
            'Drag_Adjoint`
    """

    # Initialization method
    def __init__(self, DataSet, i=None):
        """
        Initialization method for *xf_Data* interface
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
    """
    A Python class for XFlow *xf_VectorGroup* objects.
    
    :Call:
        >>> UG = xf_VectorGroup(ptr)
    
    :Parameters:
        *ptr*: :class:`int`
            Pointer to *xf_VectorGroup* struct
    
    :Data members:
        *UG._ptr*: :class:`int`
            Pointer to *xf_VectorGroup* struct
        *UG.nVector*: :class:`int`
            Number of vectors in the group
        *UG.Vector*: :class:`pyxflow.DataSet.xf_Vector` list
            List of *xf_Vector* instances
    """

    # Initialization method
    def __init__(self, ptr):
        """
        Initialization method for *xf_VectorGroup*
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
        """
        Find a vector from a vector group from its role.
        
        :Call:
            U = UG.GetVector(role="ElemState")
        
        :Parameters:
            *role*: :class:`str`
                Title of the vector to find in the vector group
        
        :Returns:
            *U*: :class:`pyxflow.DataSet.xf_Vector`
                Appropriate vector based on the role
        """
        _ptr = px.GetVectorFromGroup(self._ptr, role)
        return xf_Vector(_ptr)
        
    # Method to plot (passes information to xf_Vector.Plot())
    def Plot(self, Mesh, EqnSet, role="ElemState", **kwargs):
        """
        Plot a scalar from a vector group.
        
        :Call:
            >>> Plot = UG.Plot(Mesh, EqnSet, role="ElemState", **kwargs)
        
        :Parameters:
            *UG*: :class:`pyxflow.DataSet.xf_VectorGroup`
                Vector group containing vector to plot
            *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh for geometry data required for plotting
            *EqnSet*: :class:`pyxflow.All.xf_EqnSet`
                Equation set data
            *role*: :class:`str`
                Identifier for the vector to use for plot
                
        :Returns:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle) to use instead of creating
                a new one
            *scalar*: :class:`str`
                Name of scalar to plot
            *order*: :class:`int`
                Interpolation order for mesh faces
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
            
        :Examples:
            This can be a relatively tricky function to use, only because it
            requires the user to find a lot of the information manually.  The
            following example loads a solution file and plots the first vector
            group in the data set.
            
                >>> All = xf_All("naca_adapt_0.xfa")
                >>> UG = All.DataSet.Data[0].Data
                >>> Plot = UG.Plot(All.Mesh, All.EqnSet, xlim=[-0.5,1.5,-0.6,0.6])
            
            The result of this example is not necessarily what one might expect
            because the first vector group in ``All`` happens to be the drag
            adjoint.  Thus the above example plots the density component of the
            drag adjoint.
            
            Since the order in which vector groups are stored is not
            predictable, a more refined method is required to get the desired
            vector group.  The following will automatically find the vector
            group associated with the solution's state and plot it.
            
                >>> All = xf_All("naca_adapt_0.xfa")
                >>> UG = All.GetPrimalState()
                >>> Plot = UG.Plot(All.Mesh, All.EqnSet, xlim=[-0.5,1.5,-0.6,0.6])
                
            The *scalar* keyword argument is also particularly useful.  Any
            scalar that exists in XFlow for the active equation set can be
            plotted in this way.  For example, the following plots the pressure
            for the adapted example.  The mesh is also plotted.
            
                >>> M = All.Mesh
                >>> ES = All.EqnSet
                >>> xlim=[-0.5,1.5,-0.6,0.6] 
                >>> Plot = UG.Plot(M, ES, scalar="Pressure", xlim=xlim)
                >>> Plot = M.Plot(Plot=Plot)
                
            One convenient factor that is only subtly demonstrated in this last
            example is that by passing the plot handle to
            :func:`pyxflow.Mesh.xf_Mesh.Plot()` in the last line, there is no
            need to redefine the plot window (using *xlim* in these examples).
        
        :See also:
            :func:`pyxflow.DataSet.xf_Vector.Plot()`,
            :func:`pyxflow.Mesh.xf_Mesh.Plot()`,
            :func:`pyxflow.All.xf_All.Plot()`
        """
        # Versions:
        #  2014-02-09 @dalle   : First version
        
        # Get the vector.
        U = self.GetVector(role)
        # Plot based on the plot.
        Plot = U.Plot(Mesh, EqnSet, **kwargs)
        # Return the plot handle.
        return Plot


# ---- Class for xf_Vector ----
class xf_Vector:
    """
    A Python class for XFlow *xf_Vector* objects
    
    :Call:
        >>> U = xf_Vector(ptr)
    
    :Parameters:
        *ptr*: :class:`int`
            Pointer to *xf_Vector* struct
    
    :Data members:
        *U._ptr*: :class:`int`
            Pointer to *xf_Vector* struct
        *U.nArray*: :class:`int`
            Number of arrays in the vector
        *U.Order*: :class:`int` list
            List of interpolation order(s) used for this vector
        *U.Basis*: :class:`str` list
            List of element bases used for this vector
        *U.StateName*: :class:`str` list
            List of states in this vector
        *U.GenArray*: :class:`pyxflow.DataSet.xf_GenArray` list
            List of *U.nArray* arrays
    
    :Examples:
        If the equation set is `CompressibleNS`, the primal state's element
        state vector will have the following states.
        
            >>> All = xf_All("naca_adapt_0.xfa")
            >>> U = All.GetPrimalState().GetVector()
            >>> U.StateName
            ['Density', 'XMomentum', 'YMomentum', 'Energy']
    """

    # Initialization method
    def __init__(self, ptr):
        """
        Initialization method for *xf_Vector* interface
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
    def Plot(self, Mesh, EqnSet, scalar=None, Plot=None, **kwargs):
        """
        Plot a scalar.
        
        When *scalar* is ``None``, the first scalar available in the vector
        will be used.  This is often ``'Density'``.
        
        :Call:
            >>> Plot = U.Plot(Mesh, EqnSet, scalar=None, Plot=None, **kwargs)
            
        :Parameters:
            *U*: :class:`pyxflow.DataSet.xf_Vector`
                Vector containing scalar data to plot
            *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
                Mesh for geometry data required for plotting
            *EqnSet*: :class:`pyxflow.EqnSet.xf_EqnSet`
                Equation set data
            *scalar*: :class:`str`
                Name of scalar to plot
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle) to use instead of creating
                a new one
                
        :Returns:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            
        :Kwargs:
            *order*: :class:`int`
                Interpolation order for mesh faces
                
            See also kwargs for :func:`pyxflow.Plot.GetXLims`
        
        :See also:
            :func:`pyxflow.All.xf_All.Plot()`,
            :func:`pyxflow.DataSet.xf_VectorGroup.Plot()`,
            :func:`pyxflow.Mesh.xf_Mesh.Plot()`
        """
        # Mesh dimension
        dim = Mesh.Dim
        # Process the plot.
        if Plot is None:
            # Initialize a plot.
            Plot = pyxflow.Plot.xf_Plot()
        elif not isinstance(Plot, pyxflow.Plot.xf_Plot):
            raise IOError("Plot handle must be instance of " +
                "pyxflow.Plot.xf_Plot")
        # Use specified defaults for plot window if they exist.
        kwargs.setdefault('xmindef', Plot.xmin)
        kwargs.setdefault('xmaxdef', Plot.xmax)
        # Get the limits based on the Mesh and keyword args
        xmin, xmax = pyxflow.Plot.GetXLims(Mesh, **kwargs)
        # Save the plot limits.
        Plot.xmin = xmin
        Plot.xmax = xmax
            
        # Check for an existing mesh plot.
        if Plot.scalar is not None:
            # Delete it!
            Plot.scalar.remove()
        # Determine what figure to use.
        if kwargs.get('figure') is not None:
            # Use the input figure.
            Plot.figure = kwargs['figure']
        elif Plot.figure is None:
            # Follow norms of plotting programs; default is gcf().
            Plot.figure = plt.gcf()
        # Determine what axes to use.
        if kwargs.get('axes') is not None:
            # Use the input value.
            Plot.axes = kwargs['axes']
        else:
            # Normal plotting conventions for default
            Plot.axes = plt.gca()
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
            p = Plot.axes.tripcolor(T, scalar, shading='gouraud', cmap=colormap)
            # Store the tripcolor handle.
            Plot.scalar = p
        else:
            # Plot the value versus x.
            Plot.scalar = Plot.axes.plot(x, scalar)
        # Apply the bounding box that was created earlier.
        if kwargs.get('reset_limits', True):
            Plot.axes.set_xlim(xmin[0], xmax[0])
            if dim > 1:
                Plot.axes.set_ylim(xmin[1], xmax[1])
        # Draw if necessary.
        if plt.isinteractive():
            plt.draw()
        # Return the plot.
        return Plot


# ---- Class for xf_GenArray ----
class xf_GenArray:
    """
    A Python class for XFlow *xf_GenArray* objects
    
    This is the lowest-level class for solution, adjoint, adaptive indicator,
    etc. data.
    
    :Call:
        >>> GA = xf_GenArray(ptr)
    
    :Parameters:
        *ptr*: :class:`int`
            Pointer to *xf_GenArray* struct
    
    :Data members:
        *GA._ptr*: :class:`int`
            Pointer to *xf_GenArray* object
        *GA.n*: :class:`int`
            Number of elements represented in the array
        *GA.r*: :class:`int` or ``None``
            Number of state values per element, if constant
        *GA.vr*: :class:`numpy.array` or ``None``
            Number of state values for each element, if not constant
        *GA.rValue*: :class:`numpy.array` or ``None``
            Array of element state values if real-valued
        *GA.iValue*: :class:`numpy.array` or ``None``
            Array of element state values if integer-valued
    """

    # Initialization method
    def __init__(self, ptr):
        """
        Initialization method for *xf_GenArray*
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
