"""File to interface with XFlow xf_Geom objects in various forms"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import pyxflow._pyxflow as px


# ------- Class for xf_Geom objects -------
class xf_Geom:
    """A Python class for XFlow xf_Geom objects"""
    
    # Initialization method: 
    #  can be read from '.geom' file or existing binary object
    def __init__(self, fname=None, ptr=None):
        """
        Geom = xf_Geom(fname=None, ptr=None)
        
        INPUTS:
           fname : file name for a '.geom' file to read the geom from
           ptr   : integer pointer to existing C xf_Geom struct
        
        OUTPUTS:
           Geom  : an instance of the xf_Geom Python class
        
        This function initializes a Mesh object in one of three ways.  If
        the `fname` key is not `None`, the function will attempt to read a
        geom from a text file.  If the `ptr` is not `None`, the function
        assumes the mesh already exists on the heap and will read from it.
        If both keys are `None`, an empty mesh will be created.  Finally,
        if both keys are not `None`, an exception is raised.
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
        xf_Geom destructor
        
        This function reminds the pyxflow module to clean up the C
        xf_Geom object when the python object is deleted.
        """
        # Version:
        #  2013-09-24 @dalle   : First version
        
        if self.owner and self._ptr is not None:
            px.DestroyGeom(self._ptr)
    
    
    # Write method
    def Write(self, fname):
        """
        Geom.Write(fname)
        
        INPUTS:
           Geom  : instance of the xf_Geom class
           fname : name of geom file to write
        
        OUTPUTS:
           None
        
        This method writes an xf_Geom instance to file.
        
        """
        # Versions:
        #  2013-09-30 @dalle   : First version
        
        # Use the low-level API.
        px.WriteGeomFile(self._ptr, fname)
        # Nothing to return
        return None
        
    
    
            
# ---- Class for Geom Components ----
class xf_GeomComp:
    """A Python class for XFlow xf_GeomComp objects"""
    
    # Initialization method
    def __init__(self, Geom, i=None):
        """
        GC = xf_GeomComp(Geom, i=None)
        
        INPUTS:
           Geom : pointer from xf_Geom object
           i    : index of component to extract
        
        OUTPUTS:
           GC   : an instance of the xf_GeomComp class
        """
        # Versions:
        #  2013-09-25 @dalle   : First version
        
        # Set the initial fields.
        self.Name = None
        self.Type = None
        self.BFGTitle = None
        self.Data = None
        # Check for bad inputs.
        if Geom is None:
            return None
        
        # Read from data if appropriate
        if i is not None:
            # Fields
            self.Name, self.Type, self.BFGTitle, D = px.GeomComp(Geom, i)
            
        # Set data if possible.
        if self.Type=="Spline" and D is not None:
            # Initialize the component.
            self.Data = xf_GeomCompSpline(D["Order"],
                D["N"], D["X"], D["Y"])


# ---- Class for xf_GeomCompSline (geometry splines) ----
class xf_GeomCompSpline:
    """ A Python class for XFlow xf_GeomCompSpline objects"""
    
    # Initialization method
    def __init__(self, Order=None, N=None, X=None, Y=None):
        """
        GCS = xf_GeomCompSpline(Order, N, X, Y)
        
        INPUTS:
           Order : spline interpolation order
           N     : number of points in spline
           X     : x-coordinates of spline points
           Y     : y-coordinates of spline points
           
        OUTPUTS:
           GCS   : an instance of the xf_GeomCompSpline class
        """
        # Versions:
        #  2013-09-25 @dalle   : First version
        
        # Set the fields.
        self.Order = Order
        self.N = N
        self.X = X
        self.Y = Y







