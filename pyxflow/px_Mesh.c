#include <Python.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Mesh.h>
#include <xf_String.h>


PyObject *
px_CreateMesh(PyObject *self, PyObject *args){
  xf_Mesh *Mesh = NULL;
  const char *InputFile;
  int ierr;
	
  // Parse the input into a C string.
  if (!PyArg_ParseTuple(args, "s", &InputFile))
    return NULL;

  // Allocate the mesh.
  ierr = xf_Error(xf_CreateMesh(&Mesh));
  if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");
	
  // Read the .gri file into an xf_Mesh object.
  ierr = xf_Error(xf_ReadGriFile(InputFile, NULL, Mesh));
  if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");

  // Return the pointer
  return Py_BuildValue("n", Mesh);
}

PyObject *
px_DestroyMesh(PyObject *self, PyObject *args){
  int ierr;
  xf_Mesh *Mesh;

  PyArg_ParseTuple(args, "n", &Mesh);

  ierr = xf_Error(xf_DestroyMesh(Mesh));
  if (ierr != xf_OK) PyErr_SetString(PyExc_RuntimeError, "");

  Py_INCREF(Py_None);
  return Py_None;
}
