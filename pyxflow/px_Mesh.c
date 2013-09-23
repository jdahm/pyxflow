#include <Python.h>
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
	if (!PyArg_ParseTuple(args, "ns", &Mesh, &InputFile))
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
	if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
  
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}
