#include <Python.h>
#include "px_Mesh.h"
#include "px_Geom.h"
#include "px_DataSet.h"
#include "px_Plot.h"
#include "px_All.h"

// Need this to start NumPy C-API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#include <numpy/arrayobject.h>

#include "px_NumPy.h"


static PyMethodDef Methods[] = {
	// xf_Mesh methods
	{"CreateMesh", px_CreateMesh, METH_VARARGS,
		doc_CreateMesh},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS,
		doc_DestroyMesh},
	{"BFaceGroup", px_BFaceGroup, METH_VARARGS,
		doc_BFaceGroup},
	{"nBFaceGroup", px_nBFaceGroup, METH_VARARGS,
		doc_nBFaceGroup},
	{"ElemGroup", px_ElemGroup, METH_VARARGS,
		doc_ElemGroup},
	{"nElemGroup", px_nElemGroup, METH_VARARGS,
		doc_nElemGroup},
	{"GetNodes", px_GetNodes, METH_VARARGS,
		doc_GetNodes},
	{"ReadGriFile", px_ReadGriFile, METH_VARARGS,
		doc_ReadGriFile},
	{"WriteGriFile", px_WriteGriFile, METH_VARARGS,
		doc_WriteGriFile},
	// xf_Geom methods
	{"CreateGeom", px_CreateGeom, METH_VARARGS,
		doc_CreateGeom},
	{"DestroyGeom", px_DestroyGeom, METH_VARARGS,
		doc_DestroyGeom},
	{"nGeomComp", px_nGeomComp, METH_VARARGS,
		doc_nGeomComp},
	{"GeomComp", px_GeomComp, METH_VARARGS,
		doc_GeomComp},
	{"ReadGeomFile", px_ReadGeomFile, METH_VARARGS,
		doc_ReadGeomFile},
	{"WriteGeomFile", px_WriteGeomFile, METH_VARARGS,
		doc_WriteGeomFile},
	// xf_DataSet methods
	{"CreateDataSet", px_CreateDataSet, METH_VARARGS, 
		doc_CreateDataSet},
	{"DestroyDataSet", px_DestroyDataSet, METH_VARARGS,
		doc_DestroyDataSet},
	{"ReadDataSetFile", px_ReadDataSetFile, METH_VARARGS,
		doc_ReadDataSetFile},
	{"nDataSetData", px_nDataSetData, METH_VARARGS,
		doc_nDataSetData},
	{"GetData", px_GetData, METH_VARARGS,
		doc_GetData},
	{"GetVectorGroup", px_GetVectorGroup, METH_VARARGS,
		doc_GetVectorGroup},
	{"GetVector", px_GetVector, METH_VARARGS,
		doc_GetVector},
	{"GetVectorFromGroup", px_GetVectorFromGroup, METH_VARARGS,
		doc_GetVectorFromGroup},
	{"GetPrimalState", px_GetPrimalState, METH_VARARGS,
		doc_GetPrimalState},
	{"GetGenArray", px_GetGenArray, METH_VARARGS,
		doc_GetGenArray},
	// xf_All methods
	{"CreateAll", px_CreateAll, METH_VARARGS,
		doc_CreateAll},
	{"DestroyAll", px_DestroyAll, METH_VARARGS,
		doc_DestroyAll},
	{"ReadAllInputFile", px_ReadAllInputFile, METH_VARARGS,
		doc_ReadAllInputFile},
	{"ReadAllBinary", px_ReadAllBinary, METH_VARARGS,
		doc_ReadAllBinary},
	{"WriteAllBinary", px_WriteAllBinary, METH_VARARGS,
		doc_WriteAllBinary},
	{"GetAllMembers", px_GetAllMembers, METH_VARARGS,
		doc_GetAllMembers},
	// Plotting methods
	{"MeshPlotData", px_MeshPlotData, METH_VARARGS,
		"Creates data for plotting a mesh"},
	{"ScalarPlotData", px_ScalarPlotData, METH_VARARGS,
		"Creates data for plotting a scalar"},
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
