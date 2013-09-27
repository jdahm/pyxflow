#ifndef _PX_PLOT_H
#define _PX_PLOT_H


PyObject *
px_InterpVector2D(PyObject *self, PyObject *args);
/*
PURPOSE:
  Interpolate vector of a solution on to system of triangles
	
CALL:
  x, y, u, T = px.InterpVector2D(A, VG)
	
INPUTS:
  A  : pointer to xf_All structure
  VG : pointer to xf_VectorGroup structure

OUTPUTS:
  x : NumPy array of x-coordinates
  y : NumPy array of y-coordinates
  u : NumPy array of state values at each node
  T : NumPy array of node indices for network of triangles
*/





#endif
