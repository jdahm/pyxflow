#ifndef _PX_MESH_H
#define _PX_MESH_H

#include <Python.h>

PyObject *
px_CreateMesh(PyObject *self, PyObject *args);
/*
PURPOSE:
  Creates an empty xf_Mesh object and returns pointer

CALL:
  M = px.CreateMesh()

INPUTS:
  None

OUTPUTS:
  M : pointer to xf_Mesh
*/

PyObject *
px_ReadGriFile(PyObject *self, PyObject *args);
/*
PURPOSE:
  Reads a GRI file and returns a pointer to xf_Mesh struct

CALL:
  M = px.CreateMesh(fname)

INPUTS:
  fname : GRI file

OUTPUTS:
  M : pointer to xf_Mesh
*/

PyObject *
px_DestroyMesh(PyObject *self, PyObject *args);
/*
PURPOSE:
  Calls xf_DestroyMesh on the pointer to xf_Mesh struct

CALL:
  px.DestroyMesh(M)

INPUTS:
  M : xf_Mesh pointer

OUTPUTS:
  None
*/

#endif
