#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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
	if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
	
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
	if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
	
	// Read the .gri file into the xf_Mesh structure.
	ierr = xf_Error(xf_ReadGriFile(InputFile, NULL, Mesh));
	if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
	
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
	if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
	
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
	
	// This must be called before using the NumPy API.
	// Good luck trying to figure that out from the documentation.
	import_array();
	
	// Get the pointer to the xf_Mesh.
	PyArg_ParseTuple(args, "n", &Mesh);
	
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
	int ierr;
	
	// Get the pointer to the xf_Mesh.
	PyArg_ParseTuple(args, "n", &Mesh);
	
	// Output
	return Py_BuildValue("in", Mesh->nBFaceGroup, Mesh->BFaceGroup);
}


// Function to destroy the mesh
PyObject *
px_DestroyMesh(PyObject *self, PyObject *args)
{
	int ierr;
	xf_Mesh *Mesh;
	
	// Get the pointer.
	PyArg_ParseTuple(args, "n", &Mesh);
  
	// Deallocate the mesh.
	ierr = xf_Error(xf_DestroyMesh(Mesh));
	if (ierr != xf_OK) return NULL;
  
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}
