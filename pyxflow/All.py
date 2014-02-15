"""
The *All* module contains top-level methods for interfacing with pyXFlow's
representation of XFlow *xf_All* structs.  Its primary contents are the
:class:`pyxflow.All.xf_All` class and its members.
"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API
#  2013-09-29 @dalle   : Added a plot method
#  2013-12-15 @dalle   : First version of xf_Plot class

# ------- Modules required -------

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
    """
    Interface to XFlow *xf_EqnSet*
    
    :Call:
        >>> E = pyxflow.All.xf_EqnSet(ptr)
        
    :Parameters:
        *ptr*: :class:`int`
            Address of the *xf_EqnSet* struct
            
    :Data members:
        *E._ptr*: :class:`int`
            Pointer to XFlow *xf_EqnSet*
    """
    
    # Initialization method
    def __init__(self, ptr):
        """
        Equation set initialization method
        """
        self._ptr = ptr


class xf_All:
    """
    Interface to XFlow *xf_All*
    
    This is usually the first function used to begin with pyXFlow analysis of a
    solution.
    
    :Call:
        >>> All = pyxflow.xf_All(fname, DefaultFlag=True)
    
    :Parameters:
        *fname*: :class:`str`
            Name of file to read (usually ends in ``'.xfa'``)
        *DefaultFlag*: :class:`bool`
            Whether or not to use defaults internally
    
    :Data members:
        *All._ptr*: :class:`int`
        
        *All.Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
        
        *All.Geom*: :class:`pyxflow.Geom.xf_Geom`
        
        *All.EqnSet*: :class:`pyxflow.All.xf_EqnSet`
        
        *All.DataSet*: :class:`pyxflow.DataSet.xf_DataSet`
        
    :Examples:
        Creating an *xf_All* instance is usually straightforward.  Assuming
        that ``"naca_Adapt.xfa"`` is a file that exists on the Python path, the 
        following example loads a solution and accesses the mesh instance.
        
            >>> All = xf_All("naca_Adapt.xfa")
            ...
            >>> All.Mesh
            <pyxflow.Mesh.xf_Mesh instance at ...>
    
    """
    
    # Initialization method
    def __init__(self, fname, DefaultFlag=True):
        """
        Initialization method for :class:`pyxflow.All.xf_All`
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

    # xf_All destructor
    def __del__(self):
        """
        Deletion method for :class:`pyxflow.All.xf_All`
        """
        px.DestroyAll(self._ptr)
        
    # Method to write the xf_All back to file (after changes?)
    def Write(self, fname):
        """
        Write *xf_All* back to file.
        
        :Call:
            >>> All.Write(fname)
        
        :Parameters:
            *All*: :class:`pyxflow.All.xf_All`
                Instance of pyXFlow *xf_All* representation
            *fname*: :class:`str`
                Name of file to create
        
        :Returns:
            ``None``
        """
            

    # Method to find the primal state automatically
    def GetPrimalState(self, TimeIndex=0):
        """
        Find the primal state from an *xf_All* interface.
        
        :Call:
            >>> UG = All.GetPrimalState(TimeIndex=0)
        
        :Parameters:
            *All*: :class:`pyxflow.All.xf_All`
                Solution object from which to find primal state
            *TimeIndex*: :class:`int`
                Time index from which to find primal state
        
        :Returns:
            *UG*: :class:`pyxflow.DataSet.xf_VectorGroup`
                Vector group for the primal state
        """
        ptr = px.GetPrimalState(self._ptr, TimeIndex)
        return xf_VectorGroup(ptr)

    # Master plotting method
    def Plot(self, scalar=None, **kwargs):
        """
        Plot the mesh and scalar from *xf_All* representation
        
        :Call:
            >>> plot = All.Plot(scalar=None, **kwargs)

        :Parameters:
            *All*: :class:`pyxflow.All.xf_All`
                Instance of the pyXFlow *xf_All* interface
            *scalar*: :class:`str`
                Name of scalar to plot
                
                A value of ``None`` uses the default scalar.
                
                A value of ``False`` prevents plotting of any scalar.

        :Returns:
            *plot*: :class:`pyxflow.Plot.xf_Plot`
                pyXFlow plot instance with mesh and scalar handles
        
        :Kwargs:
            *mesh*: :class:`bool`
                Whether or not to plot the mesh
            *plot*: :class:`pyxflow.Plot.xf_Plot`
                Instance of plot class (plot handle)
            *role*: :class:`str`
                Identifier for the vector to use for plot
                The default value is ``'ElemState'``.
            *order*: :class:`int`
                Interpolation order for mesh faces
            *vgroup*: :class:`pyxflow.DataSet.xf_VectorGroup`
                Vector group to use for plot
                
                A value of ``None`` results in using the primal state.
                
                The behavior of this keyword argument is subject to change.
                
            See also kwargs for 
                * :func:`pyxflow.Plot.GetXLims()`
                * :func:`pyxflow.Mesh.xf_Mesh.Plot()`
                * :func:`pyxflow.DataSet.xf_Vector.Plot()`
            
        :Examples:
            The following example loads an airfoil solution and plots the
            pressure near the surface.
            
                >>> All = xf_All("naca_Adapt.xfa")
                ...
                >>> All.Plot("Pressure", xlim=[-0.9,1.2,-0.5,0.6])
                <pyxflow.Mesh.xf_Mesh instance at ...>
            
        :See also:
            * :func:`pyxflow.Mesh.xf_Mesh.Plot()`
            * :func:`pyxflow.DataSet.xf_Vector.Plot()`
            * :func:`pyxflow.DataSet.xf_VectorGroup.Plot()`
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


