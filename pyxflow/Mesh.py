"""File to interface with XFlow mesh objects in various forms"""

# Versions:
#  2013-06-11 @dalle   : First version


# ------- Modules required -------
# Used for more efficient data storage
import numpy as np
# The background pyxflow workhorse module
import _pyxflow as px

# ------- CLASSES -------
# --- Class to represent the (full) mesh ---
class xf_Mesh:
    """A Python class for XFlow mesh objects"""
    
    # Parameters
    pointer = None
    Dim = 0
    nNode = 0
    Coord = None
    nIFace = 0
    IFace = None
    nBFaceGroup = 0
    BFaceGroup = None
    nElemGroup = 0
    ElemGroup = None
    nPeriodicGroup = 0
    PeriodicGroup = None
    ParallelInfo = None
    BackgroundMesh = None
    Motion = None
    
    # Method to initialize the object
    def __init__(self, gri=None, ptr=None):
        """
        Mesh = xf_Mesh(gri=None, ptr=None)
        
        INPUTS:
           gri  : file name for a '.gri' file to read the mesh from
           ptr  : integer pointer to existing C xf_Mesh struct
        
        OUTPUTS:
           Mesh : an instance of the xf_Mesh Python class
        
        This function initializes a Mesh object in one of three ways.  If
        the `gri` key is not `None`, the function will attempt to read a
        mesh from a text file.  If the `ptr` is not `None`, the function
        assumes the mesh already exists on the heap and will read from it.
        If both keys are `None`, an empty mesh will be created.  Finally,
        if both keys are not `None`, an exception is raised.
        """
        
        # Versions:
        #  2013-09-23 @dalle   : First version
        
        # Check the parameters.
        if gri is not None:
            if ptr is not None:
                raise NameError
            # Read the file and get the pointer.
            ptr = px.ReadGriFile(gri)
            # Set it.
            self.pointer = ptr
	    self.owner = True
        elif ptr is not None:
            # Simply set the pointer.
            self.pointer = ptr
	    self.owner = False
        else:
            # Create an empty mesh.
            ptr = px.CreateMesh()
            # Set the pointer.
            self.pointer = ptr
            self.owner = True

    # Destructor method for xf_Mesh
    def __del__(self):
        """
        xf_Mesh destructor
        
        This function reminds the pyxflow module to clean up the C
        xf_Mesh object when the python object is deleted.
        """
        
        # Version:
        #  2013-09-23 @dalle   : First version
        
        # Always have a mesh to destroy
        if self.owner:
            px.DestroyMesh(self.pointer)



# --- Class just for meshes read from '.gri' files ---
class xf_GriFile:
    """GRI file class for pyXFlow, a Python interface for XFlow"""
    # Parameters
    nNode = 0
    Dim = 0
    Coord = None
    nBFaceGroup = 0
    BFaceGroup = None
    nElemGroup = 0
    ElemGroup = None
    nElemTot = 0
    
    
    # Function to read in meshes from file.
    @classmethod
    def Read(cls, fname):
        """
        Mesh = xf_GriFile.Read(fname)
        
        INPUTS:
           fname : name of file to be read
        
        OUTPUTS:
           Mesh  : xf_GriFile object
        
        This function reads a '.gri' file into a Python representation
        of an abbreviated XFlow mesh.  It only has the properties to describe
        a mesh compatible with '.gri' files.

        >>> M = xf_GriFile.Read("../examples/uniform_tri_q1_2.gri")
        """
        
        # Versions:
        #  2013-09-20 @dalle   : First version
        
        # Initialize the object.
        Mesh = cls()
        
        # Open the file.
        f = open(fname, 'r')
        
        # Read first line.
        l = f.readline()
        # Split first line into numbers.
        D = l.split()
        
        # Number of nodes
        Mesh.nNode = int(D[0])
        # Total number of elements not part of mesh object
        Mesh.nElemTot = int(D[1])
        # Number of dimensions in the grid.
        Mesh.Dim = int(D[2])
        
        # Initialize a variable for the mesh.
        R = np.zeros((Mesh.nNode, Mesh.Dim))
        # Read nNode lines
        for i in range(Mesh.nNode):
            # Read a line and split it.
            D = f.readline().split()
            # Save the coordinates
            for j in range(Mesh.Dim):
                R[i,j] = float(D[j])
        # Save the coordinates.
        Mesh.Coord = R
        
        # Read line for number of boundary condition groups
        D = f.readline().split()
        # Only one number
        Mesh.nBFaceGroup = int(D[0])
        # Initialize boundary groups
        BG = []
        # Loop through the groups
        for i in range(Mesh.nBFaceGroup):
            # Read a line describing the group.
            D = f.readline().split()
            # Number of faces
            nBFace = int(D[0])
            # Number of linear nodes per face
            nf = int(D[1])
            # Initialize a boundary group.
            BG.append(xf_BGroup(D[2], nBFace, nf))
            # Loop through the faces.
            for j in np.arange(nBFace):
                # Read a line describing the nodes in the face.
                B = f.readline().split()
                # Save the entries
                BG[i].NB[j,0] = int(B[0])
                BG[i].NB[j,1] = int(B[1])
        # Save the boundary face groups.
        Mesh.BFaceGroup = BG
        
        # Running total number of elements
        nElemCur = 0
        # Running total number of of element groups
        nElemGroup = 0
        # Initialize the element group object
        EG = []
        # Read until the number of elements equals the total
        while nElemCur < Mesh.nElemTot:
            # Read line for number of element groups
            D = f.readline().split()
            # Check for end of file.
            if len(D) == 0:
                break
            # Number of elements
            nElem = int(D[0])
            # Add to the running total.
            nElemCur += nElem
            # Order
            iOrder = int(D[1])
            # Calculate the number of nodes per element
            if D[2].lower() == "quadlagrange":
                # quadrilateral elements
                nn = (iOrder + 1)**Mesh.Dim
            elif D[2].lower() == "trilagrange":
                # Check dimension
                if Mesh.Dim == 2:
                    # Triangular elements
                    nn = (iOrder+1) * (iOrder+2) / 2
                else:
                    # Tetrahedral elements
                    nn = (iOrder+1)*(iOrder+2)*(iOrder+3) / 6
            else:
                f.close()
                raise NameError('Unrecognized basis type.')
            # Initialize an element group
            EG.append(xf_EGroup(nElem, iOrder, D[2], nn))
            # Loop through the lines.
            for j in np.arange(nElem):
                # Read the line.
                B = f.readline().split()
                # Loop through the elements.
                for k in range(nn):
                    EG[nElemGroup].NE[j,k] = int(B[k])
            # Add to the number of element groups.
            nElemGroup += 1
        # Save the number of element groups
        Mesh.nElemGroup = nElemGroup
        # Save the element groups.
        Mesh.ElemGroup = EG
        
        # Close the file.
        f.close()
        # Output the mesh.
        return Mesh
    
    
    # === Method to write '.gri' files ===
    def Write(self, fname):
        """
        Mesh.Write(fname)
        
        INPUTS:
           Mesh  : xf_GriFile object
           fname : name of file to write to
        
        OUTPUTS:
           (None)
        
        This method writes a '.gri' file using the contents of an
        xf_GriFile object.
        """
        # Versions:
        #  2013-09-20 @dalle   : First version
        
        # Open the file for writing.
        f = open(fname, 'w')
        
        # Write the first line.
        # nNode nElemTot Dim
        f.write("%i %i %i\n" % (self.nNode, self.nElemTot, self.Dim))
        
        # Loop through the nodes.
        for i in np.arange(self.nNode):
            # Loop through the dimensions
            for j in range(self.Dim):
                # Print the coordinates.
                f.write("%.15e " % (self.Coord[i,j]))
            # Print a newline character.
            f.write("\n")
        
        # Write the number of boundary face groups.
        f.write(str(self.nBFaceGroup) + "\n")
        # Loop through the boundary face groups.
        for i in range(self.nBFaceGroup):
            # Extract the group.
            BG = self.BFaceGroup[i]
            # Write the header for the boundary face group.
            f.write("%i %i %s\n" % (BG.nBFace, BG.nf, BG.Title))
            # Loop through the face elements.
            for j in np.arange(BG.nBFace):
                # Loop through the nodes in the element.
                # This is where my Python coding is clearly inadequate.
                for k in range(BG.nf):
                    f.write("%i " % (BG.NB[j,k]))
                # Move to the next line.
                f.write("\n")
        
        # Loop through the element groups.
        for i in range(self.nElemGroup):
            # Extract the group.
            EG = self.ElemGroup[i]
            # Write the header line for group i.
            f.write("%i %i %s\n" % (EG.nElem, EG.Order, EG.Basis))
            # Loop through the elements.
            for j in np.arange(EG.nElem):
                # Loop through the nodes. :(
                for k in range(EG.nn):
                    # Print the number
                    f.write("%i " % (EG.NE[j,k]))
                # Move to the next line.
                f.write("\n")
                
        # Close the file.
        f.close()
        # End function
        return None
 
 
# --- Class for .gri file boundary face groups ---
class xf_BGroup:
    """
    GRI-file boundary face group for pyXFlow, a Python interface for XFlow
    """
    # Initialization method
    def __init__(self, Title='', nBFace=0, nf=0):
        # Set the parameters
        self.Title  = Title
        self.nBFace = nBFace
        self.nf     = nf
        # Initialize the node numbers
        self.NB = np.zeros((nBFace, nf))
    # It would be nice to define a conversion method here...
    #    xf_BFaceGroup --> xf_BGroup


# --- Class for .gri file boundary face groups ---
class xf_EGroup:
    """
    GRI-file element group for pyXFlow, a Python interface for XFlow
    """
    # Initialization method
    def __init__(self, nElem=0, Order=0, Basis='', nn=0):
        # Set the known parameters
        self.nElem = nElem
        self.Order = Order
        self.Basis = Basis
        self.nn    = nn
        # Initialize the node set
        self.NE = np.zeros((nElem, nn))
    # It would be nice to define a conversion method here...
    #    xf_BFaceGroup --> xf_BGroup



# --- Class for boundary face groups ---
class xf_BFaceGroup:
    """
    Boundary face group object for pyxflow, a Python interface for XFlow
    
    """
    # Initialization method
    def __init__(self, Title='', nBFace=0, BFace=[]):
        # Define the properties
        self.Title  = Title
        self.nBFace = nBFace
        self.BFace  = BFace
    # No methods for now
    # It would be nice to define a conversion method here...
    #    xf_BGroup --> xf_BFaceGroup


# --- Class for boundary faces ---
class xf_BFace:
    """
    Boundary face class for pyxflow, a Python interface for XFlow
    
    """
    # Initialization method
    def __init__(self, ElemGroup=0, Elem=0, Face=0, Orient=0):
        # Define the properties.
        self.ElemGroup = ElemGroup
        self.Elem      = Elem
        self.Face      = Face
        self.Orient    = Orient
    # No methods for now
    
