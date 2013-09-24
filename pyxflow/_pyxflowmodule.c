#include <Python.h>
#include "px_Mesh.h"

// Steps to import the NumPy API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#include <numpy/arrayobject.h>

static PyMethodDef Methods[] = {
	{"CreateMesh",  px_CreateMesh,  METH_VARARGS, "Create empty xf_Mesh"},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS, "Destroy mesh and free memory"},
	{"BFaceGroup",  px_BFaceGroup,  METH_VARARGS, "Get BFaceGroup info"},
	{"nBFaceGroup", px_nBFaceGroup, METH_VARARGS, "Get BFaceGroup pointer"},
	{"ElemGroup",   px_ElemGroup,   METH_VARARGS, "Get ElemGroup info"},
	{"nElemGroup",  px_nElemGroup,  METH_VARARGS, "Get ElemGroup pointer"},
	{"GetNodes",    px_GetNodes,    METH_VARARGS, "Get node coordinate info"},
	{"ReadGriFile", px_ReadGriFile, METH_VARARGS, "Read GRI file to xf_Mesh"},
	{"WriteGriFile",px_WriteGriFile,METH_VARARGS, "Write xf_Mesh to GRI file"},
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
