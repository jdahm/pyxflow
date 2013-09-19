# File to interface with XFlow mesh objects in various forms

class xf_Mesh:
    """A Python class for XFlow mesh objects"""
    
    # Parameters
    nDim = 0
    nNode = 0
    Coord = {}
    nIFace = 0
    IFace = {}
    nBFaceGroup = 0
    BFaceGroup = {}
    nElemGroup = 0
    ElemGroup = {}
    nPeriodicGroup = 0
    PeriodicGroup = {}
    ParallelInfo = {}
    BackgroundMesh = {}
    Motion = {}
    
    # Function to read in meshes from file.
    @classmethod
    def ReadGriFile(cls, fname):
        """
        Mesh = xf_Mesh.ReadGriFile(fname)
        
        INPUTS:
           fname : name of file to be read
        
        OUTPUTS:
           Mesh  : xf_Mesh object
        
        This function reads a '.gri' file into a Python representation
        of an XFlow mesh.  It is constructed to mimic the properties of
        the native 'xf_Mesh' object in C.
        """
        
        # Versions:
        #  2013-06-11 @dalle   : First version
        #
        # Aliases:
        #  @dalle   : Derek J. Dalle <dalle@umich.edu>
        
        # Initialize the object.
        Mesh = cls();
        
        return Mesh
