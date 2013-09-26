#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Data.h>



// Function to create an empty geom.
PyObject *
px_CreateDataSet(PyObject *self, PyObject *args)
{
	xf_DataSet *DataSet = NULL;
	int ierr;

	// Allocate the mesh.
	ierr = xf_Error(xf_CreateDataSet(&DataSet));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", DataSet);
}


// Function to read xf_DataSet from (?) '.data' file
PyObject *
px_ReadDataSetFile(PyObject *self, PyObject *args)
{
	xf_DataSet *DataSet = NULL;
	xf_Mesh *Mesh = NULL;
	const char *fname;
	int ierr;
	
	// Parse the python inputs.
	if (!PyArg_ParseTuple(args, "ns", &Mesh, &fname))
		return NULL;
	
	// Allocate the DataSet.
	ierr = xf_Error(xf_CreateDataSet(&DataSet));
	if (ierr != xf_OK) return NULL;
	
	// Read the file into the xf_DataSet structure.
	ierr = xf_Error(xf_ReadDataSetBinary(Mesh, NULL, fname, DataSet));
	if (ierr != xf_OK) return NULL;
	
	// Return the pointer.
	return Py_BuildValue("n", DataSet);
}


// Function to get the number of data chunks in the xf_DataSet
PyObject *
px_nDataSetData(PyObject *self, PyObject *args)
{
	xf_DataSet *DataSet;
	xf_Data *D;
	int nData = 0;
	
	// Parse the Python inputs.
	if (!PyArg_ParseTuple(args, "n", &DataSet))
		return NULL;
	
	// Initialize the xf_Data pointer.
	D = DataSet->Head;
	
	// Loop until the tail is found (or an error).
	while (D != NULL)
	{
		// Increase the Data count.
		nData++;
		// Check if it's the last data set.
		if (D == DataSet->Tail) break;
	}
	// Check if an error was found.
	if (D == NULL) return NULL;
	
	// Return the number of entries.
	return Py_BuildValue("i", nData);
}



// Function to destroy the mesh
PyObject *
px_DestroyDataSet(PyObject *self, PyObject *args)
{
	int ierr;
	xf_DataSet *DataSet;
	
	// Get the pointer.
	if (!PyArg_ParseTuple(args, "n", &DataSet)) return NULL;
  
	// Deallocate the mesh.
	ierr = xf_Error(xf_DestroyDataSet(DataSet));
	if (ierr != xf_OK) return NULL;
  
	// Nothing to return.
	Py_INCREF(Py_None);
	return Py_None;
}
