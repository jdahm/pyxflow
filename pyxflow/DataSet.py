"""File to interface with XFlow xf_DataSet objects in various forms"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import pyxflow._pyxflow as px


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
    
    
    # xf_DataSet destructor method
    def __del__(self):
        """
        xf_DataSet destructor
        
        This function reminds the pyxflow module to clean up the C
        xf_DataSet object when the Python object is deleted.
        """
        # Version:
        #  2013-09-25 @dalle   : First version
        
        if self._ptr is not None:
            px.DestroyDataSet(self._ptr)
        







