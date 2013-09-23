#include <Python.h>
#include "px_Mesh.h"

static PyMethodDef Methods[] = {
	{"CreateMesh",  px_CreateMesh,  METH_VARARGS, "Create empty xf_Mesh"},
	{"ReadGriFile", px_ReadGriFile, METH_VARARGS, "Read GRI file to xf_Mesh"},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS, "Destroy mesh and free memory"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_pyxflow(void)
{
	(void) Py_InitModule("_pyxflow", Methods);
}
