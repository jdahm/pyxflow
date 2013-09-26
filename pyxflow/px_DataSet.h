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
