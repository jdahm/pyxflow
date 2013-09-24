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
px_WriteGriFile(PyObject *self, PyObject *args);
/*
PURPOSE:
  Write an xf_Mesh object to a GRI file

CALL:
  px.WriteGriFile(M, fname)

INPUTS:
  M     : xf_Mesh pointer (python int)
  fname : file name for output
  
OUTPUTS:
  None
*/


PyObject *
px_GetNodes(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get basic nodes information from xf_Mesh object
  
CALL:
  (Dim, nNode, Coord) = px.GetNodes(M)

INPUTS:
  M : xf_Mesh pointer
  
OUTPUTS:
  Dim   : number of dimensions [ 2 | 3 ]
  nNode : number of nodes
  Coord : numpy.ndarray of coordinates  [(nNode)x(Dim) ndarray]
*/


PyObject *
px_nBFaceGroup(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get number of boundary conditions and corresponding pointer
  
CALL:
  (nBFaceGroup, BFG) = px.nBFaceGroup(M)

INPUTS:
  M : xf_Mesh pointer

OUTPUTS:
  nBFaceGroup : number of BFaceGroup structs in xf_Mesh
  BFG         : pointer to Mesh->BFaceGroup
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
