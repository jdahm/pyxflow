#ifndef _PX_DATASET_H
#define _PX_DATASET_H

PyObject *
px_CreateDataSet(PyObject *self, PyObject *args);
char doc_CreateDataSet[] =
"Create an empty *xf_DataSet* instance.\n"
"\n"
":Call:\n"
"   >>> DS = px.CreateDataSet()\n"
"\n"
":Parameters:\n"
"   ``None``\n"
"\n"
":Returns:\n"
"   *DS*: :class:`int`\n"
"       Pointer to empty *xf_DataSet* instance\n";


PyObject *
px_ReadDataSetFile(PyObject *self, PyObject *args);
char doc_ReadDataSetFile[] =
"Create an *xf_DataSet* instance from a `.data` file.\n"
"\n"
":Call:\n"
"   >>> DS = px.ReadDataSetFile(M, fname)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* struct that the data set corresponds to\n"
"   *fname*: :class:`str`\n"
"       Name of data file to read\n"
"\n"
":Returns:\n"
"   *DS*: :class:`int`\n"
"       Pointer to *xf_DataSet* instance\n";


PyObject *
px_nDataSetData(PyObject *self, PyObject *args);
char doc_nDataSetData[] =
"Get the number of data set components from an *xf_DataSet* instance.\n"
"\n"
":Call:\n"
"   >>> nData = px.nDataSetData(DS)\n"
"\n"
":Parameters:\n"
"   *DS*: :class:`int`\n"
"       Pointer to *xf_DataSet* instance\n"
"\n"
":Returns:\n"
"   *nData*: :class:`int`\n"
"       Number of data components\n";


PyObject *
px_GetData(PyObject *self, PyObject *args);
char doc_GetData[] =
"Get types and titles of each *xf_Data* in an *xf_DataSet*\n"
"\n"
":Call:\n"
"   >>> Title, Type, D, Data = px.GetData(DS, iData)\n"
"\n"
":Parameters:\n"
"   *DS*: :class;`int`\n"
"       Pointer to *xf_DataSet*\n"
"   *iData*: :class:`int`\n"
"       Index of the *xf_Data* to use, starting form *DataSet->D->Head*\n"
"\n"
":Returns:\n"
"   *Title*: :class:`str`\n"
"       Name of the data component\n"
"   *Type*: :class:`str`\n"
"       Type of data component, usually ``'VectorGroup'`` or ``'Vector'``\n"
"   *D*: :class:`int`\n"
"       Pointer to *xf_Data* (*DataSet->D*)\n"
"   *Data*: :class:`int`\n"
"       Pointer to vector, vector group, etc. (*DataSet->D->Data*)\n";


PyObject *
px_GetVectorGroup(PyObject *self, PyObject *args);
char doc_GetVectorGroup[] =
"Get pointer to vectors in *xf_VectorGroup*\n"
"\n"
":Call:\n"
"   >>> nVector, U = px.GetVectorGroup(UG)\n"
"\n"
":Parameters:\n"
"   *UG*: :class:`int`\n"
"       Pointer to *xf_VectorGroup*\n"
"\n"
":Returns:\n"
"   *nVector*: :class:`int`\n"
"       Number of vectors in the group\n"
"   *U*: :class:`int` list\n"
"       List of pointers to *xf_Vector* instances\n";


PyObject *
px_GetVector(PyObject *self, PyObject *args);
char doc_GetVector[] =
"Get information about *xf_Vector* from its pointer\n"
"\n"
":Call:\n"
"   >>> nArray, Order, Basis, StateName, GA = px.GetVector(U)\n"
"\n"
":Parameters:\n"
"   *U*: :class:`int`\n"
"       Pointer to *xf_Vector*\n"
"\n"
":Returns:\n"
"   *nArray*: :class:`int`\n"
"       Number of *xf_GenArray* instances in the vector\n"
"   *Order*: :class:`int`\n"
"       Interpolation order for the vector\n"
"   *Basis*: :class:`str`\n"
"       Name of basis used for the elements in this vector\n"
"   *StateName*: :class:`str` list\n"
"       List of states used in this vector\n"
"   *GA*: :class:`int` list\n"
"       List of pointers to *xf_GenArray* instances\n";


PyObject *
px_GetVectorFromGroup(PyObject *self, PyObject *args);
char doc_GetVectorFromGroup[] =
"Return an *xf_Vector* pointer from a group based on the vector's role\n"
"\n"
":Call:\n"
"   >>> U = px.GetVector(UG, URole)\n"
"\n"
":Parameters:\n"
"   *UG*: :class:`int`\n"
"       Pointer to *xf_VectorGroup* struct\n"
"   *URole*: :class:`str`\n"
"       Vector role string\n"
"\n"
":Returns:\n"
"   *U*: :class:`int`\n"
"       Pointer to requested *xf_Vector*\n";

PyObject *
px_GetPrimalState(PyObject *self, PyObject *args);
char doc_GetPrimalState[] =
"Return a pointer to the primal state vector group\n"
"\n"
":Call:\n"
"   >>> UG = px.GetPrimalState(A, TimeIndex)\n"
"\n"
":Parameters:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All* struct\n"
"   *TimeIndex* :class:`int`\n"
"       Time step index\n"
"\n"
":Returns:\n"
"   *UG*: :class:`int`\n"
"       Pointer to appropriate *xf_VectorGroup*\n";


PyObject *
px_GetGenArray(PyObject *self, PyObject *args);
char doc_GetGenArray[] =
"Get data from *xf_GenArray* struct\n"
"\n"
":Call:\n"
"   >>> D = px.GetGenArray(GA)\n"
"\n"
":Parameters:\n"
"   *GA*: :class:`int`\n"
"       Pointer to *xf_GenArray* struct\n"
"\n"
":Returns:\n"
"   *D*: :class:`dict`\n"
"       Data from the *xf_GenArray*\n"
"\n"
":See also:\n"
"   :func:`pyxflow.DataSet.xf_GenArray`";



PyObject *
px_DestroyDataSet(PyObject *self, PyObject *args);
char doc_DestroyDataSet[] =
"Destroy an *xf_DataSet* instance and free memory.\n"
"\n"
":Call:\n"
"   >>> px.DestroyDataSet(DS)\n"
"\n"
":Parameters:\n"
"   *DS*: :class:`int`\n"
"       Pointer to *xf_DataSet* instance\n"
"\n"
":Returns:\n"
"   ``None``\n";

#endif
