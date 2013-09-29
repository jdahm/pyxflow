#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Mesh.h>
#include <xf_String.h>



// Function to create an empty mesh.
PyObject *
px_CreateMesh(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	int ierr;

	// Allocate the mesh.
	ierr = xf_Error(xf_CreateMesh(&Mesh));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", Mesh);
}


// Function to read mesh from .gri file
PyObject *
px_ReadGriFile(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	const char *InputFile;
	int ierr;
	
	// Parse the python inputs.
	if (!PyArg_ParseTuple(args, "s", &InputFile))
		return NULL;
	
	// Allocate the mesh.
	ierr = xf_Error(xf_CreateMesh(&Mesh));
	if (ierr != xf_OK) return NULL;
	
	// Read the .gri file into the xf_Mesh structure.
	ierr = xf_Error(xf_ReadGriFile(InputFile, NULL, Mesh));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", Mesh);
}


// Function to write xf_Mesh to .gri file
PyObject *
px_WriteGriFile(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	char *OutputFile;
	int ierr;
	
	// Parse the python inputs.
	if (!PyArg_ParseTuple(args, "ns", &Mesh, &OutputFile))
		return NULL;
	
	// Write the mesh to a .gri file.
	ierr = xf_Error(xf_WriteGriFile(Mesh, OutputFile));
	if (ierr != xf_OK) return NULL;
	
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}


// Function to extract the node information
PyObject *
px_GetNodes(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	PyObject *np_Coord;
	npy_intp dims[2];
	
	// Get the pointer to the xf_Mesh.
	if (!PyArg_ParseTuple(args, "n", &Mesh))
		return NULL;
	
	// Get dimensions
	dims[0] = Mesh->nNode;
	dims[1] = Mesh->Dim;
	
	// Error checking
	if (Mesh->nNode <= 0) return NULL;
	
	// Make the mesh.
	np_Coord = PyArray_SimpleNewFromData( \
		2, dims, NPY_DOUBLE, *Mesh->Coord);
	
	// Output (Dim, nNode, Coord).
	return Py_BuildValue("iiO", Mesh->Dim, Mesh->nNode, np_Coord);
}


// Function to extract the boundary conditions
PyObject *
px_nBFaceGroup(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	
	// Get the pointer to the xf_Mesh.
	if (!PyArg_ParseTuple(args, "n", &Mesh))
		return NULL;
	
	// Output
	return Py_BuildValue("i", Mesh->nBFaceGroup);
}


// Function to read the BFaceGroup
PyObject *
px_BFaceGroup(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	xf_BFaceGroup *BFG = NULL;
	int iBFG, nBFace;
	const char * Title;
	
	// Get the pointer to the xf_BFaceGroup
	if (!PyArg_ParseTuple(args, "ni", &Mesh, &iBFG))
		return NULL;
	
	// Get the BFaceGroup
	BFG = Mesh->BFaceGroup;

	// Read the title
	Title = BFG[iBFG].Title;
	nBFace = BFG[iBFG].nBFace;
	
	// Output: (Title, nBFace, _BFace[0])
	return Py_BuildValue("sin", Title, nBFace, &BFG[iBFG]);
}


// Function to extract element group information
PyObject *
px_nElemGroup(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	
	// Get the pointer to the xf_Mesh.
	if (!PyArg_ParseTuple(args, "n", &Mesh))
		return NULL;
	
	// Output
	return Py_BuildValue("i", Mesh->nElemGroup);
}


// Function to read the BFaceGroup
PyObject *
px_ElemGroup(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	xf_ElemGroup *EG = NULL;
	int i, nElem, nNode, QOrder;
	const char *QBasis;
	PyObject *Node;
	npy_intp dims[2];
	
	// Get the pointer to the xf_BFaceGroup
	if (!PyArg_ParseTuple(args, "ni", &Mesh, &i))
		return NULL;
	
	// Assign the element group
	EG = Mesh->ElemGroup;
	
	// Read the data
	nElem  = EG[i].nElem;
	nNode  = EG[i].nNode;
	QOrder = EG[i].QOrder;
	// Convert the enumeration to a string.
	QBasis = xfe_BasisName[EG[i].QBasis];
	
	// Dimensions of the nodes array
	dims[0] = nElem;
	dims[1] = nNode;
	// Read the nodes to a numpy array.
	Node = PyArray_SimpleNewFromData( \
		2, dims, NPY_INT, *EG[i].Node);
	
	// Output: (nElem, nNode, QOrder, QBasis, Node)
	return Py_BuildValue("iiisO", nElem, nNode, QOrder, QBasis, Node);
}


// Function to destroy the mesh
PyObject *
px_DestroyMesh(PyObject *self, PyObject *args)
{
	int ierr;
	xf_Mesh *Mesh;
	
	// Get the pointer.
	if (!PyArg_ParseTuple(args, "n", &Mesh)) return NULL;
  
	// Deallocate the mesh.
	ierr = xf_Error(xf_DestroyMesh(Mesh));
	if (ierr != xf_OK) return NULL;
  
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}

