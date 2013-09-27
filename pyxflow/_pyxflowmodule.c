#include <Python.h>
#include "px_Mesh.h"
#include "px_Geom.h"
#include "px_DataSet.h"
#include "px_Plot.h"
#include "px_All.h"

// Steps to import the NumPy API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#include <numpy/arrayobject.h>

static PyMethodDef Methods[] = {
	// xf_Mesh methods
	{"CreateMesh", px_CreateMesh, METH_VARARGS,
		"Create empty xf_Mesh"},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS,
		"Destroy mesh and free memory"},
	{"BFaceGroup", px_BFaceGroup, METH_VARARGS,
		"Get BFaceGroup info"},
	{"nBFaceGroup", px_nBFaceGroup, METH_VARARGS,
		"Get BFaceGroup pointer"},
	{"ElemGroup", px_ElemGroup, METH_VARARGS,
		"Get ElemGroup info"},
	{"nElemGroup", px_nElemGroup, METH_VARARGS,
		"Get ElemGroup pointer"},
	{"GetNodes", px_GetNodes, METH_VARARGS,
		"Get node coordinate info"},
	{"ReadGriFile", px_ReadGriFile, METH_VARARGS,
		"Read GRI file to xf_Mesh"},
	{"WriteGriFile", px_WriteGriFile, METH_VARARGS,
		"Write xf_Mesh to GRI file"},
	// xf_Geom methods
	{"CreateGeom", px_CreateGeom, METH_VARARGS,
		"Create empty xf_Geom"},
	{"DestroyGeom", px_DestroyGeom, METH_VARARGS,
		"Destroy geom and free memory"},
	{"nGeomComp", px_nGeomComp, METH_VARARGS,
		"Get Geom->Comp pointer and Geom->nComp"},
	{"GeomComp", px_GeomComp, METH_VARARGS,
		"Get data from Geom->Comp"},
	{"ReadGeomFile", px_ReadGeomFile, METH_VARARGS,
		"Read '.geom' file to xf_Geom"},
	// xf_DataSet methods
	{"CreateDataSet", px_CreateDataSet, METH_VARARGS, 
		"Create empty xf_DataSet"},
	{"DestroyDataSet", px_DestroyDataSet, METH_VARARGS,
		"Destroy xf_DataSet and free memory"},
	{"ReadDataSetFile", px_ReadDataSetFile, METH_VARARGS,
		"Read '.data' file to xf_DataSet (requires xf_Mesh)"},
	{"nDataSetData", px_nDataSetData, METH_VARARGS,
		"Get number of xf_Data structs in xf_DataSet"},
	{"GetData", px_GetData, METH_VARARGS,
		"Get information about ith xf_Data in a DataSet"},
	{"GetVectorGroup", px_GetVectorGroup, METH_VARARGS,
		"Get pointers to vectors from xf_VectorGroup"},
	{"GetVector", px_GetVector, METH_VARARGS,
		"Get information on xf_Vector from pointer"},
	{"GetGenArray", px_GetGenArray, METH_VARARGS,
		"Get information from an xf_GenArray pointer"},
	// xf_All methods
	{"CreateAll", px_CreateAll, METH_VARARGS,
		"Create empty xf_All"},
	{"DestroyAll", px_DestroyAll, METH_VARARGS,
		"Destroy all and free memory"},
	{"ReadAllInputFile", px_ReadAllInputFile, METH_VARARGS,
		"Read all from input file"},
	{"WriteAllBinary", px_WriteAllBinary, METH_VARARGS,
		"Writes all to binary file"},
	{"GetAllMembers", px_GetAllMembers, METH_VARARGS,
		"Returns a tuple of pointers to the members of the xf_All"},
	// Plotting methods
	{"InterpVector2D", px_InterpVector2D, METH_VARARGS,
		"Creates network of triangles and values for plotting"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_pyxflow(void)
{
	// This must be called before using the NumPy API.
	import_array();
	// Initialization command.
	(void) Py_InitModule("_pyxflow", Methods);
}
