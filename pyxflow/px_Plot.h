#ifndef _PX_PLOT_H
#define _PX_PLOT_H


PyObject *
px_MeshPlotData(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Calculates mesh data for plotting

    CALL:
    x, y, con = px.MeshPlotData(M, min, max)

    INPUTS:
    M   : pointer to xf_Mesh structure
    min : [xmin, (ymin, zmin)]
    max : [xmax, (ymax, zmax)]

    OUTPUTS:
    x   : x-position of nodes [np]
    y   : y-position of nodes [np]
    con : connectivity data.  The positions are stored in {x,y}[C[f],
        C[f+1]] for some face index f
*/

PyObject *
px_ScalarPlotData(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Calculates mesh data for plotting

    CALL:
    x, y, tri, scalar = px.MeshPlotData(U, M, E, Name, min, max)

    INPUTS:
    U    : pointer to xf_Vector structure
    M    : pointer to xf_Mesh structure
    E    : pointer to xf_EqnSet structure
    Name : name of scalar
           if NULL and interpolated : interpolated and plots the
             first component of the vector
           if not interpolated : plots first entry in vector for
             each element
    min  : [xmin, (ymin, zmin)]
    max  : [xmax, (ymax, zmax)]

    OUTPUTS:
    x      : x-position of nodes [np]
    y      : y-position of nodes [np]
    tri    : node indices for each triangle [np,3]
    scalar : scalar value at each node
*/


#endif
