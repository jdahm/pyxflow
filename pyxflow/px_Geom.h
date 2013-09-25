#ifndef _PX_GEOM_H
#define _PX_GEOM_H

PyObject *
px_CreateGeom(PyObject *self, PyObject *args);
/*
PURPOSE:
  Creates an empty xf_Geom object and returns pointer

CALL:
  G = px.CreateGeom()

INPUTS:
  None

OUTPUTS:
  G : pointer to xf_Geom
*/


PyObject *
px_ReadGeomFile(PyObject *self, PyObject *args);
/*
PURPOSE:
  Read a '.geom' file and create an xf_Geom structure
  
CALL:
  G = px.ReadGeomFile(fname)
  
INPUTS:
  fname : name of "geom" file
  
OUTPUTS:
  G : pointer to xf_Geom
*/





PyObject *
px_DestroyGeom(PyObject *self, PyObject *args);
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
