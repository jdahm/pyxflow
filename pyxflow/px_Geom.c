#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Geom.h>
#include <xf_GeomIO.h>
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


// Function to extract the number of geom components
PyObject *
px_nGeomComp(PyObject *self, PyObject *args)
{
	xf_Geom *Geom = NULL;
	
	// Get the pointer to the xf_Mesh.
	if (!PyArg_ParseTuple(args, "n", &Geom))
		return NULL;
	
	// Output
	return Py_BuildValue("i", Geom->nComp);
}


// Function to read the Geom->Comp information
PyObject *
px_GeomComp(PyObject *self, PyObject *args)
{
	xf_Geom *Geom;
	xf_GeomComp *GC;
	xf_GeomCompSpline *GCS;
	PyObject *D;
	int iComp;
	char *Name, *BFGTitle, *Type;
	
	// Get the pointer to the xf_BFaceGroup
	if (!PyArg_ParseTuple(args, "ni", &Geom, &iComp))
		return NULL;
	
	// Check the value of iComp.
	if (iComp >= Geom->nComp){
		PyErr_SetString(PyExc_RuntimeError, \
			"Component index exceeds dimensions.");
		return NULL;
	}
	
	// Pointer to component
	GC = Geom->Comp+iComp;
	
	// Read the name.
	Name = GC->Name;
	// Convert Type to string.
	Type = xfe_GeomCompName[GC->Type];
	// Name of corresponding boundary condition
	BFGTitle = GC->BFGTitle;
	
	// Determine the component type
	switch(Geom->Comp[iComp].Type){
	case xfe_GeomCompSpline:
		// Get the spline pointer.
		GCS = (xf_GeomCompSpline *) GC->Data;
		// Get the spline interpolation order.
		int Order = GCS->Order;
		// Number of points
		int N = GCS->N;
		npy_intp dims[1] = {N};
		// coordinates
		PyObject *X = PyArray_SimpleNewFromData( \
			1, dims, NPY_DOUBLE, GCS->X);
		PyObject *Y = PyArray_SimpleNewFromData( \
			1, dims, NPY_DOUBLE, GCS->Y);
		// Create the object
		D = Py_BuildValue("{sisisOsO}", "Order", Order, "N", N, "X", X, "Y", Y);
		break;
	default:
		// Return `None` for the data
		Py_INCREF(Py_None);
		D = Py_None;
		break;
	}
	
	// Output: (Title, nBFace, _BFace[0])
	return Py_BuildValue("sssO", Name, Type, BFGTitle, D);
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

