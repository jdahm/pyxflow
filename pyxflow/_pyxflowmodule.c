#include <Python.h>
#include "px_Mesh.h"
#include "px_All.h"

static PyMethodDef Methods[] = {
	{"CreateMesh",       px_CreateMesh,       METH_VARARGS, "Create empty xf_Mesh"},
	{"DestroyMesh",      px_DestroyMesh,      METH_VARARGS, "Destroy mesh and free memory"},
	{"GetNodes",         px_GetNodes,         METH_VARARGS, "Get node coordinate info"},
	{"ReadGriFile",      px_ReadGriFile,      METH_VARARGS, "Read GRI file to xf_Mesh"},
	{"WriteGriFile",     px_WriteGriFile,     METH_VARARGS, "Write xf_Mesh to GRI file"},
	{"CreateAll",        px_CreateAll,        METH_VARARGS, "Create empty xf_All"},
	{"DestroyAll",       px_DestroyAll,       METH_VARARGS, "Destroy all and free memory"},
	{"ReadAllInputFile", px_ReadAllInputFile, METH_VARARGS, "Read all from input file"},
	{"WriteAllBinary",   px_WriteAllBinary,   METH_VARARGS, "Writes all to binary file"},
	{"GetAllMembers",    px_GetAllMembers,   METH_VARARGS, "Returns a tuple of pointers to the members of the xf_All"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_pyxflow(void)
{
	(void) Py_InitModule("_pyxflow", Methods);
}
