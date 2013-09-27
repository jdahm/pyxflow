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
	while (D != NULL && nData < 100)
	{
		// Increase the Data count.
		nData++;
		// Check if it's the last data set.
		if (D == DataSet->Tail) break;
		// Move to next DataSet.
		D = D->Next;
	}
	
	// Return the number of entries.
	return Py_BuildValue("i", nData);
}


// Function to read info from the xf_Data
PyObject *
px_GetData(PyObject *self, PyObject *args)
{
	xf_DataSet *DataSet;
	xf_Data *D;
	xf_VectorGroup *VG;
	PyObject *pV;
	char *Title, *Type;
	int i, n, iData;

	// Parse the Python inputs.
	if (!PyArg_ParseTuple(args, "ni", &DataSet, &iData))
		return NULL;
	
	// Initialize the xf_Data pointer.
	D = DataSet->Head;
	
	// Loop until the correct data has been found.
	for (i=0; i<iData; i++){
		D = D->Next;
	}
	
	// Read the title and type.
	Title = D->Title;
	Type  = xfe_DataName[D->Type];
	
	// Outputs : (Title, Type, ptr_Data).
	return Py_BuildValue("ssnn", Title, Type, D, D->Data);
}



// Function to read an xf_VectorGroup from an xf_Data
PyObject *
px_GetVectorGroup(PyObject *self, PyObject *args)
{
	xf_VectorGroup *VG;
	PyObject *V;
	int i, nVector;
	
	// Parse the Python inputs.
	if (!PyArg_ParseTuple(args, "n", &VG))
		return NULL;
	
	// Get the number of vectors.
	nVector = VG->nVector;
	// Make a list of pointers to xf_Vector objects.
	V = PyList_New((Py_ssize_t) nVector);
	// Loop through the vectors to get the pointers.
	for (i=0; i<nVector; i++) {
		// Set the pointer to V[i].
		PyList_SetItem(V, (Py_ssize_t) i, Py_BuildValue("n", VG->Vector[i]));
	}
	
	// Output: number of vectors and list of pointers
	return Py_BuildValue("nO", nVector, V);
}


// Function to read a vector from a pointer
PyObject *
px_GetVector(PyObject *self, PyObject *args)
{
	xf_Vector *V;
	PyObject *GA, *Order, *Basis, *StateName;
	const char *UBasis;
	int i;
	int nArray, StateRank;
	Py_ssize_t j;
	
	// Parse the Python inputs.
	if (!PyArg_ParseTuple(args, "n", &V))
		return NULL;
	
	// Get the number of arrays.
	nArray = V->nArray;
	
	// Make a list of pointers to xf_Vector objects.
	GA = PyList_New((Py_ssize_t) nArray);
	// Lists of orders and bases
	Order = PyList_New((Py_ssize_t) nArray);
	Basis = PyList_New((Py_ssize_t) nArray);
	// Loop through the GenArrays
	for (i=0; i<nArray; i++) {
		// Set the pointer to V->GenArray[i].
		PyList_SetItem(GA, i, Py_BuildValue("n", V->GenArray+i));
		
		// Set the order.
		if (V->Order != NULL) {
			// Use the value.
			PyList_SetItem(Order, i, Py_BuildValue("i", V->Order[i]));
		} else {
			// Empty Order
			Py_INCREF(Py_None);
			PyList_SetItem(Order, i, Py_None);
		}
		
		// Store the Basis.
		if (V->Basis != NULL) {
			// Basis name
			UBasis = xfe_BasisName[V->Basis[i]];
			// Use the value.
			PyList_SetItem(Basis, i, Py_BuildValue("s", UBasis));
		} else {
			// Empty Basis field
			Py_INCREF(Py_None);
			PyList_SetItem(Basis, i, Py_None);
		}
	}
	
	// Check the StateRank
	if (V->StateRank == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "Vector state rank is NULL.");
		return NULL;
	}
	// StateRank
	StateRank = V->StateRank;
	// Set the StateNames if appropriate.
	if (V->StateName != NULL) {
		// Initialize list of StateNames.
		StateName = PyList_New((Py_ssize_t) StateRank);
		// Loop through the states.
		for (i=0; i<StateRank; i++) {
			// Use the value.
			PyList_SetItem(StateName, i, Py_BuildValue("s", V->StateName[i]));
		}
	} else {
		// Empty StateName field
		Py_INCREF(Py_None);
		StateName = Py_None;
	}
	
	// Output
	return Py_BuildValue("iOOOO", nArray, Order, Basis, StateName, GA);
}


// Function to read a GenArray from a pointer
PyObject *
px_GetGenArray(PyObject *self, PyObject *args)
{
	xf_GenArray *G;
	PyObject *vr, *iValue, *rValue;
	int i, n, r;
	npy_intp dims1[1], dims2[2];
	
	
	// Get the pointer from Python.
	if (!PyArg_ParseTuple(args, "n", &G))
		return NULL;
	
	// Integer properties
	n = G->n; r = G->r;
	
	// Check for variable size.
	if (G->vr != NULL) {
		// Dimensions
		dims1[0] = n;
		// Number of entries per row
		vr = PyArray_SimpleNewFromData(1, dims1, NPY_INT, G->vr);
	}
	else {
		// Make a reference to None.
		Py_INCREF(Py_None);
		vr = Py_None;
	}
	
	// Check for integer data
	if (G->iValue != NULL) {
		// Check for variable size.
		if (G->vr != NULL) {
			// Variable dimensions: list of arrays
			iValue = PyList_New(n);
			// Loop through the elements.
			for (i=0; i<n; i++) {
				// Set the dimension.
				dims1[0] = G->vr[i];
				// Assign an array to the ith element of the list.
				PyList_SetItem(iValue, i, PyArray_SimpleNewFromData( \
					1, dims1, NPY_INT, G->iValue[i]));
			}
		}
		else {
			// Constant dimension: large array
			// Rectangular dimensions
			dims2[0] = n;
			dims2[1] = r;
			// Assign the data
			iValue = PyArray_SimpleNewFromData(2, dims2, NPY_INT, *G->iValue);
		}
	}
	else {
		// Make a reference to None.
		Py_INCREF(Py_None);
		iValue = Py_None;
	}
	
	// Check for real data
	if (G->rValue != NULL) {
		// Check for variable size.
		if (G->vr != NULL) {
			// Variable dimensions: list of arrays
			iValue = PyList_New(n);
			// Loop through the elements.
			for (i=0; i<n; i++) {
				// Set the dimension.
				dims1[0] = G->vr[i];
				// Assign an array to the ith element of the list.
				PyList_SetItem(iValue, i, PyArray_SimpleNewFromData( \
					1, dims1, NPY_DOUBLE, G->rValue[i]));
			}
		}
		else {
			// Constant dimension: large array
			// Rectangular dimensions
			dims2[0] = n;
			dims2[1] = r;
			// Assign the data
			rValue = PyArray_SimpleNewFromData(2, dims2, NPY_DOUBLE, *G->rValue);
		}
	}
	else {
		// Make a reference to None.
		Py_INCREF(Py_None);
		rValue = Py_None;
	}
	
	// Return a dictionary.
	return Py_BuildValue("{sisisOsOsO}", "n", n, "r", r, "vr", vr, \
		"iValue", iValue, "rValue", rValue);
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
