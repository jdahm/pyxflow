#ifndef _PX_PLOT_H
#define _PX_PLOT_H


PyObject *
px_InterpVector(PyObject *self, PyObject *args);
/*
PURPOSE:
  Interpolate vector of a solution on to system of triangles
	
CALL:
  x, y, u, T = px.InterpVector(A, VG)
	
INPUTS:
  A  : pointer to xf_All structure
  VG : pointer to xf_VectorGroup structure

OUTPUTS:
  x : NumPy array of x-coordinates
  y : NumPy array of y-coordinates
  u : NumPy array of state values at each node
  T : NumPy array of node indices for network of triangles
*/


PyObject *
px_GetRefineCoords(PyObject *self, PyObject *args);
/*
PURPOSE:
  Wrapper for xf_GetRefineCoords

CALL:
  coord, vsplit, vbound =  px.GetRefineCoords(Shape, p)

INPUTS:
  Shape : shape reference integer
  p     : order

OUTPUTS:
  coord  : 2D array of reference coordinates
  vsplit : nodes of vertices of tris/tets
  vbound : indices of boundary nodes
*/




#endif
