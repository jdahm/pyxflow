#include <Python.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Mesh.h>
#include <xf_String.h>

static PyObject *
px_Mesh_Add(PyObject *self, PyObject *args)
{
	int x, y;

	if (!PyArg_ParseTuple(args, "ii", &x, &y))
		return NULL;
	return Py_BuildValue("i", x+y);
}

static PyObject *
px_ReadGriFile(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	const char *InputFile;
	int ierr;
	int nNode;
	
	// Parse the input into a C string.
	if (!PyArg_ParseTuple(args, "s", &InputFile))
		return NULL;
	
	// Allocate the mesh.
	ierr = xf_Error(xf_CreateMesh(&Mesh));
	if (ierr != xf_OK) return NULL;
	
	// Read the .gri file into an xf_Mesh object.
	ierr = xf_Error(xf_ReadGriFile(InputFile, NULL, Mesh));
	if (ierr != xf_OK) return NULL;
	
	printf("%p\n%d\n", Mesh, Mesh);
	
	// Return the number of elements
	return Py_BuildValue("i", Mesh);
}

static PyObject *
px_nNode(PyObject *self, PyObject *args)
{
	xf_Mesh *Mesh = NULL;
	int pMesh = 0;
	int ierr;
	int nNode;
	
	// Parse the input into a pointer.
	if (!PyArg_ParseTuple(args, "i", &pMesh))
		return NULL;
	
	printf("%d\n", pMesh);
	
	// Get the Mesh from the pointer.
	Mesh = pMesh;
	
	// Get the number of nodes.
	nNode = Mesh->nNode;
	
	return Py_BuildValue("i", nNode);
}


// Definitions for px_Mesh methods
static PyMethodDef MeshMethods[] = {
	{"Add",  px_Mesh_Add,    METH_VARARGS, "Add two numbers... but in C"},
	{"Read", px_ReadGriFile, METH_VARARGS, "Number of nodes from GRI"},
	{"nNode", px_nNode,      METH_VARARGS, "Get number of nodes"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initpx_Mesh(void)
{
	(void) Py_InitModule("px_Mesh", MeshMethods);
}
