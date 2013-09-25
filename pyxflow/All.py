"""File with Top-level interfaces to XFlow"""

# Versions:
#  2013-09-18 @jdahm   : First version
#  2013-09-23 @jdahm   : Integrated C-API

# ------- Modules required -------
# Used for parsing input from files
import re
# The background pyxflow workhorse module
import _pyxflow as px
# Mesh
from pyxflow.Mesh import xf_Mesh
# Geom
from pyxflow.Geom import xf_Geom

class xf_DataSet:
    pass

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
        (Mesh_ptr, Geom_ptr, DataSet_ptr, Param_ptr, EqnSet_ptr) = px.GetAllMembers(self._ptr)

        # Shadow the members inside this class
        self.Mesh   = xf_Mesh(ptr=Mesh_ptr)
        self.EqnSet = xf_EqnSet(EqnSet_ptr)

    def __del__(self):
        px.DestroyAll(self._ptr)

