#include <Python.h>
#include "px_Mesh.h"

static PyMethodDef Methods[] = {
	{"CreateMesh", px_CreateMesh, METH_VARARGS, "Create and read xf_Mesh from GRI file"},
	{"DestroyMesh", px_DestroyMesh, METH_VARARGS, "Destroy mesh and free memory"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initpythonxflow(void)
{
	(void) Py_InitModule("pythonxflow", Methods);
}
