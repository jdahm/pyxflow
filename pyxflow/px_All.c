#include <Python.h>
#include "xf_AllStruct.h"
#include "xf_All.h"

PyObject *
px_CreateAll(PyObject *self, PyObject *args)
{
	int ierr;
	enum xfe_Bool DefaultFlag;
        xf_All *All = NULL;

	if (!PyArg_ParseTuple(args, "b", DefaultFlag)) return NULL;

	// Allocate the xf_All struct
	ierr = xf_Error(xf_CreateAll(&All, DefaultFlag));
	if (ierr != xf_OK) return NULL;

	// Return the pointer
	return Py_BuildValue("n", All);
}

PyObject *
px_DestroyAll(PyObject *self, PyObject *args)
{
	int ierr;
        xf_All *All = NULL;

	if (!PyArg_ParseTuple(args, "n", &All)) return NULL;

	// Destroy the xf_All struct
	ierr = xf_Error(xf_DestroyAll(All));
	if (ierr != xf_OK) return NULL;

	// Nothing to return
	Py_INCREF(Py_None);
	return Py_None;
}

PyObject *
px_ReadAllInputFile(PyObject *self, PyObject *args)
{
	int ierr;
	char *InputFile;
	enum xfe_Bool DefaultFlag;
        xf_All *All = NULL;

	if (!PyArg_ParseTuple(args, "sb", &InputFile, &DefaultFlag)) return NULL;

	// Create and read in the xf_All struct from file
	ierr = xf_Error(xf_ReadAllInputFile(InputFile, NULL, DefaultFlag, &All));
	if (ierr != xf_OK) return NULL;

	// Return the pointer
	return Py_BuildValue("n", All);
}

PyObject *
px_WriteAllBinary(PyObject *self, PyObject *args)
{
	int ierr;
        xf_All *All = NULL;
	char *fname;

	if (!PyArg_ParseTuple(args, "ns", &All, &fname)) return NULL;

	// Write xf_All struct to file
	ierr = xf_Error(xf_WriteAllBinary(All, fname));
	if (ierr != xf_OK) return NULL;

	// Nothing to return
	Py_INCREF(Py_None);
	return Py_None;
}

PyObject *
px_GetAllMembers(PyObject *self, PyObject *args)
{
        xf_All *All = NULL;

	if (!PyArg_ParseTuple(args, "n", &All)) return NULL;

	// Return a tuple of pointers
	return Py_BuildValue("nnnnn", All->Mesh, All->Geom, All->DataSet, All->Param, All->EqnSet);
}
