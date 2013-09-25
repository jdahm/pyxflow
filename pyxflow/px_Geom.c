#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Geom.h>
#include <xf_String.h>


// Function to create an empty geom.
PyObject *
px_CreateGeom(PyObject *self, PyObject *args)
{
	xf_Geom *Geom = NULL;
	int ierr;

	// Allocate the mesh.
	ierr = xf_Error(xf_CreateGeom(&Geom));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", Geom);
}


// Function to read mesh from .gri file
PyObject *
px_ReadGeomFile(PyObject *self, PyObject *args)
{
	xf_Geom *Geom = NULL;
	const char *fname;
	int ierr;
	
	// Parse the python inputs.
	if (!PyArg_ParseTuple(args, "s", &fname))
		return NULL;
	
	// Allocate the mesh.
	ierr = xf_Error(xf_CreateGeom(&Geom));
	if (ierr != xf_OK) return NULL;
	
	// Read the .gri file into the xf_Mesh structure.
	ierr = xf_Error(xf_ReadGeomFile(fname, NULL, Geom));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", Geom);
}





// Function to destroy the mesh
PyObject *
px_DestroyGeom(PyObject *self, PyObject *args)
{
	int ierr;
	xf_Geom *Geom;
	
	// Get the pointer.
	if (!PyArg_ParseTuple(args, "n", &Geom)) return NULL;
  
	// Deallocate the mesh.
	ierr = xf_Error(xf_DestroyGeom(Geom));
	if (ierr != xf_OK) return NULL;
  
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}

