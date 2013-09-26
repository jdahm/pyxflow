#ifndef _PX_DATASET_H
#define _PX_DATASET_H

PyObject *
px_CreateDataSet(PyObject *self, PyObject *args);
/*
PURPOSE:
  Creates an empty xf_DataSet object and returns pointer

CALL:
  DS = px.CreateDataSet()

INPUTS:
  None

OUTPUTS:
  DS : pointer to xf_DataSet
*/


PyObject *
px_ReadDataSetFile(PyObject *self, PyObject *args);
/*
PURPOSE:
  Read a '.data' file to xf_DataSet
  
CALL:
  DS = px.ReadDataSetFile(M, fname)
  
INPUTS:
  M     : pointer to an xf_Mesh struct
  fname : name of data file
  
OUTPUTS:
  DS : pointer to xf_DataSet struct
*/


PyObject *
px_nDataSetData(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get number of data entries in an xf_DataSet
  
CALL:
  nData = px.nDataSetData(DS)
  
INPUTS:
  DS : pointer to xf_DataSet struct
  
OUTPUTS:
  nData : number of xf_Data structs in linked list
*/


PyObject *
px_GetData(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get types and titles of (and pointers to) each xf_Data in a DataSet

CALL:
  Title, Type, D, Data = px.GetData(DS, iData)
  
INPUTS:
  DS    : pointer to xf_DataSet struct
  iData : index of xf_Data to use
  
OUTPUTS:
  Title : title of xf_Data struct
  Type  : type of xf_Data being presented
  D     : pointer to xf_Data (DataSet->D)
  Data  : pointer to vector, vector group, etc. (DataSet->D->Data)
*/


PyObject *
px_GetVectorGroup(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get number of and pointers to vectors in xf_VectorGroup
  
CALL:
  nVector, V = px.GetVectorGroup(VG)
  
INPUTS:
  VG : pointer to xf_VectorGroup struct
  
OUTPUTS:
  nVector : number of vectors present
  V       : list of pointers to vectors
*/


PyObject *
px_GetVector(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get information about xf_Vector from pointer
  
CALL:
  nArray, Order, Basis, StateName, GA = px.GetVector(V)
  
INPUTS:
  V : pointer to xf_Vector struct
  
OUTPUTS:
  nArray    : number of arrays in vector
  Order     : interpolation order for arrays
  Basis     : name of basis for arrays
  StateName : name of state described by arrays
  GA        : list of pointers to GenArrays
*/


PyObject *
px_GetGenArray(PyObject *self, PyObject *args);
/*
PURPOSE:
	Get data from xf_GenArray structure
	
CALL:
	D = px.GetGenArray(GA)
	
INPUTS:
	GA : pointer to xf_GenArray struct

OUTPUTS:
	D : dictionary of xf_GenArray information

NOTES:
  (1) The following parameters are defined for the dictionary. 
      
        D
         ["n"]      : number of elements
         ["r"]      : number of entries per element
         ["vr"]     : variable number of degrees of freedom for each elem
         ["iValue"] : integer data on the array
         ["rValue"] : real-valued data on the array
         
      Depending on the GenArray, some of the values may be `None`
*/




PyObject *
px_DestroyDataSet(PyObject *self, PyObject *args);
/*
PURPOSE:
  Calls xf_DestroyGeom on the pointer to xf_Geom struct

CALL:
  px.DestroyGeom(G)

INPUTS:
  G : xf_Geom pointer

OUTPUTS:
  None
*/

#endif
