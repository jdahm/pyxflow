"""File with Top-level interfaces to XFlow"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API

# ------- Modules required -------
# Used for parsing input from files
import re
# Matplotlit essentials
import matplotlib.pyplot as plt
# The background pyxflow workhorse module
import _pyxflow as px
# Mesh
from pyxflow.Mesh import xf_Mesh
# Geom
from pyxflow.Geom import xf_Geom
# DataSet
from pyxflow.DataSet import xf_DataSet




class xf_Param:
    pass

class xf_EqnSet:

    def __init__(self, ptr):
        self._ptr = ptr


class xf_All:

    def __init__(self, fname, DefaultFlag=True):

        # Create an xf_All instance in memory
        self._ptr = px.ReadAllInputFile(fname, True)

        # Get pointers to all members
        (Mesh_ptr, Geom_ptr, DataSet_ptr, Param_ptr, 
            EqnSet_ptr) = px.GetAllMembers(self._ptr)

        # Shadow the members inside this class
        self.Mesh    = xf_Mesh(ptr=Mesh_ptr)
        self.Geom    = xf_Geom(ptr=Geom_ptr)
        self.EqnSet  = xf_EqnSet(EqnSet_ptr)
        self.DataSet = xf_DataSet(ptr=DataSet_ptr)
        

    def __del__(self):
        px.DestroyAll(self._ptr)
        
    def Plot(self):
        """
        All = xf_All(...)
        All.Plot()
        
        INPUTS:
        
        
        OUTPUTS:
        
        
        This is the plotting method for the xf_All class.  More capabilities
        will be added.
        """
        # Versions:
        #  2013-09-29 @dalle   : First version
        
        # Check for a DataSet
        if not (self.DataSet.nData >= 1):
            return None
        
        # Check that we have a vector group.
        if not (self.DataSet.Data[0].Type == 'VectorGroup'):
            return None
            
        # This is for 2D right now!
        if self.Mesh.Dim != 2:
            return None
        
        # Get the vector group.
        UG = self.DataSet.Data[0].Data
        
        # Get the data and triangulation.
        X, u, T = px.InterpVector2D(self._ptr, UG._ptr)
        
        # Process.....
        
        
        # Draw the plot
        # Using density for now.
        h = plt.tripcolor(X[:,0], X[:,1], T, u[:,0], shading='gouraud')
        
        # return the handle
        return h
        
        
        

