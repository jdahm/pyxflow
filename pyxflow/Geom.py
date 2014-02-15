"""
The *Geom* module contains the primary interface for XFlow's *xf_Geom* structs,
which describe the boundaries and surfaces in XFlow.  The "geom" of a boundary
is not the same as the boundary condition because, among other reasons, surfaces
should be defined more accurately than the boundary elements.  Otherwise, if a
boundary element is refined, the new surface node may not lie on the intended
surface.

Included in this module are several methods to access internal *xf_Geom*
functions.  In other words, it provides an API for XFlow geom methods.  In
addition, this module enables interactive or scripted surface deformation by
shifting the node locations and rewriting the geometry to file.
"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import pyxflow._pyxflow as px


# ------- Class for xf_Geom objects -------
class xf_Geom:
    """
    A Python class for XFlow *xf_Geom* objects
    
    If an instance is created with a pointer and no file name, the geom is
    processed from the existing *xf_Geom* struct (through the XFlow API).
    
    If both the file name and the pointer are ``None``, an empty *xf_Geom* is
    created.
    
    :Call:
        >>> Geom = pyxflow.xf_Geom(fname=None, ptr=None)
    
    :Parameters:
        *fname*: :class:`str`
            Name of *.geom* file to read the geometry from
        *ptr*: :class:`int`
            Pointer to existing *xf_Geom* struct
    
    :Data members:
        *Geom._ptr*: :class:`int`
            Pointer to *xf_Geom* instance
        *Geom.nComp*: :class:`int`
            Number of geometry components in the geometry description
        *Geom.Comp*: :class:`pyxflow.Geom.xf_GeomComp` list
            List of *xf_GeomComp* interfaces
    """

    # Initialization method:
    #  can be read from '.geom' file or existing binary object
    def __init__(self, fname=None, ptr=None):
        """
        Initialization method for *xf_Geom*
        """
        # Versions:
        #  2013-09-25 @dalle   : First version

        # Check the parameters.
        if fname is not None:
            if ptr is not None:
                raise NameError
            # Read the file and get the pointer.
            ptr = px.ReadGeomFile(fname)
            # Set it.
            self._ptr = ptr
            self.owner = True
        elif ptr is not None:
            # Set the pointer.
            self._ptr = ptr
            self.owner = False
        else:
            # Create an empty geom.
            ptr = px.CreateGeom()
            # Set the pointer
            self._ptr = ptr
            self.owner = True
            # Exit the function
            return None

        # Read the number of components.
        self.nComp = px.nGeomComp(self._ptr)
        # Initialize the components
        self.Comp = [xf_GeomComp(self._ptr, i) for i in range(self.nComp)]

    # Destructor method for xf_Mesh
    def __del__(self):
        """
        Destructor for *xf_Geom*
        """
        # Version:
        #  2013-09-24 @dalle   : First version

        if self.owner and self._ptr is not None:
            px.DestroyGeom(self._ptr)

    # Write method
    def Write(self, fname):
        """
        Write an *xf_Geom* struct to file.
        
        :Call:
            >>> Geom.Write(fname)
        
        :Parameters:
            *Geom*: :class:`pyxflow.Geom.xf_Geom`
                Object containing geometry information to save
            *fname*: :class:`str`
                Name of file to write
        
        :Returns:
            ``None``
        """
        # Versions:
        #  2013-09-30 @dalle   : First version

        # Use the low-level API.
        px.WriteGeomFile(self._ptr, fname)
        # Nothing to return
        return None


# ---- Class for Geom Components ----
class xf_GeomComp:
    """
    A Python class for XFlow *xf_GeomComp* objects
    
    :Call:
        >>> GC = pyxflow.Geom.xf_GeomComp(ptr, i=None)
    
    :Parameters:
        *ptr*: :class:`int`
            Pointer to *xf_Geom*
        *i*: :class:`int`
            Index of geometry component to read
    
    :Data members:
        *GC.Name*: :class:`str`
            User-set name of the geometry component
        *GC.Type*: :class:`str`
            Type of geometry component, often ``'Spline'``
        *GC.BFGTitle*: :class:`str`
            Name of boundary face group to which component applies
        *GC.Data*: :class:`pyxflow.Geom.xf_GeomCompSpline` or ``None``
            Spline data if appropriate
    """

    # Initialization method
    def __init__(self, ptr, i=None):
        """
        Initialization method for *xf_GeomComp*
        """
        # Versions:
        #  2013-09-25 @dalle   : First version

        # Set the initial fields.
        self.Name = None
        self.Type = None
        self.BFGTitle = None
        self.Data = None
        # Check for bad inputs.
        if ptr is None:
            return None

        # Read from data if appropriate
        if i is not None:
            # Fields
            self.Name, self.Type, self.BFGTitle, D = px.GeomComp(ptr, i)

        # Set data if possible.
        if self.Type == "Spline" and D is not None:
            # Initialize the component.
            self.Data = xf_GeomCompSpline(D["Order"],
                                          D["N"], D["X"], D["Y"])


# ---- Class for xf_GeomCompSline (geometry splines) ----
class xf_GeomCompSpline:
    """
    A Python class for XFlow *xf_GeomCompSpline* objects
    
    This class has data members identical to the parameters to its
    initialization method.
    
    :Call:
        >>> GCS = xf_GeomCompSpline(Order, N, X, Y)
    
    :Parameters:
        *Order*: :class:`int`
            Spline interpolation order
        *N*: :class:`int`
            Number of points in spline
        *X*: :class:`numpy.array` (*N*)
            *x*-coordinates of spline points
        *Y*: :class:`numpy.array` (*N*)
            *y*-coordinates of spline points
    """

    # Initialization method
    def __init__(self, Order=None, N=None, X=None, Y=None):
        """
        Initialization method for *xf_GeomCompSpline*
        """
        # Versions:
        #  2013-09-25 @dalle   : First version

        # Set the fields.
        self.Order = Order
        self.N = N
        self.X = X
        self.Y = Y

