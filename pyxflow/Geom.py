"""File to interface with XFlow mesh objects in various forms"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import pyxflow._pyxflow as px


# ------- Class for xf_Geom objects -------
class xf_Geom:
    """A Python class for XFlow mesh objects"""
    
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
            
            
    # Destructor method for xf_Mesh
    def __del__(self):
        """
        xf_Geom destructor
        
        This function reminds the pyxflow module to clean up the C
        xf_Geom object when the python object is deleted.
        """
        # Version:
        #  2013-09-24 @dalle   : First version
        
        if self._ptr is not None:
            px.DestroyGeom(self._ptr)
            
            
            
            
