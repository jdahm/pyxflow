#ifndef _PX_PLOT_H
#define _PX_PLOT_H


PyObject *
px_PlotData(PyObject *self, PyObject *args);
/*
PURPOSE:
  Calculate interpolated vector and mesh line coordinates
	
CALL:
  X, u, T, L = px.InterpVector(A, UG, xlim)
	
INPUTS:
  A    : pointer to xf_All structure
  UG   : pointer to xf_VectorGroup structure
  xlim : plot window, [xmin, xmax, ymin, ymax(, zmin, zmax)]

OUTPUTS:
  X : NumPy array of spatial node coordinates
  u : NumPy array of state values at each node
  T : NumPy array of node indices for network of triangles
  L : list of arrays of boundary nodes
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
