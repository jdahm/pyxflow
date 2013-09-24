#include <Python.h>
#include "px_Mesh.h"

static PyMethodDef Methods[] = {
	{"CreateMesh",  px_CreateMesh,  METH_VARARGS, "Create empty xf_Mesh"},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS, "Destroy mesh and free memory"},
	{"GetNodes",    px_GetNodes,    METH_VARARGS, "Get node coordinate info"},
	{"nBFaceGroup", px_nBFaceGroup, METH_VARARGS, "Get BFaceGroup pointer"},
	{"ReadGriFile", px_ReadGriFile, METH_VARARGS, "Read GRI file to xf_Mesh"},
	{"WriteGriFile",px_WriteGriFile,METH_VARARGS, "Write xf_Mesh to GRI file"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_pyxflow(void)
{
	(void) Py_InitModule("_pyxflow", Methods);
}
