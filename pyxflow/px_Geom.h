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
px_nGeomComp(PyObject *self, PyObject *args);
/*
PURPOSE:
  Get Geom->nComp and pointer to Geom->Comp
	
CALL:
  nComp, C = px.nGeomComp(G)
	
INPUTS:
  G : pointer to xf_Geom struct

OUTPUTS:
  nComp : number of geometry components
  C     : pointer to Geom->Comp
*/


PyObject *
px_GeomComp(PyObject *self, PyObject *args);
/*
PURPOSE:
	Get data from xf_GeomComp structure
	
CALL:
	Name, Type, BFGTitle, D = px.GeomComp(G, iComp)
	
INPUTS:
	G     : pointer to xf_Geom struct
	iComp : index of component to analyze

OUTPUTS:
	Name     : name of component
	Type     : type of component, see xfe_GeomCompName
	BFGTitle : title of corresponding boundary condition
	D        : data from the component (Note 1)

NOTES:
  (1) If `Type` is "Spline", a dict with the following values will be returned. 
      
        D
         ["Order"] : interpolation order of the spline
         ["N"]     : number of points in the spline
         ["X"]     : x-coordinates of points
         ["Y"]     : y-coordinates of points
         
      If the value of `Type` is not spline, the value of `D` will be `None`.
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
