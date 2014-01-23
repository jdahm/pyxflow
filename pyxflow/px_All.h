#ifndef _PX_ALL_H
#define _PX_ALL_H

PyObject *
px_CreateAll(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Creates an empty xf_All object and returns pointer

    CALL:
    A = px.CreateAll(DefaultFlag)

    INPUTS:
    DefaultFlag : true to set default parameters

    OUTPUTS:
    A : pointer to empty xf_All
*/

PyObject *
px_DestroyAll(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Calls xf_DestroyMesh on the pointer to xf_All struct

    CALL:
    px.DestroyAll(A)

    INPUTS:
    A : xf_All pointer

    OUTPUTS:
    None
*/

PyObject *
px_ReadAllInputFile(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Creates an xf_All struct by calling xf_ReadAllInputFile with an
    input file.

    CALL:
    A = px.ReadAllInputFile(InputFile, DefaultFlag)

    INPUTS:
    InputFile   : Input filename
    DefaultFlag : true to set default parameters

    OUTPUTS:
    A : xf_All pointer
*/

PyObject *
px_ReadAllBinary(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Creates an xf_All struct by calling xf_ReadAllInputFile with an
    input file.

    CALL:
    A = px.ReadAllBinary(XfaFile, DefaultFlag)

    INPUTS:
    XfaFile   : Input filename
    DefaultFlag : true to set default parameters

    OUTPUTS:
    A : xf_All pointer
*/

PyObject *
px_WriteAllBinary(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Writes the xf_All content to file.

    CALL:
    px.WriteAllBinary(A, fname)

    INPUTS:
    A     : xf_All struct
    fname : Output filename

    OUTPUTS:
    None
*/

PyObject *
px_GetAllMembers(PyObject *self, PyObject *args);
/*
    PURPOSE:
    Returns member pointers.

    CALL:
    (Mesh, Geom, DataSet, Param, EqnSet) = px.GetAllMembers(A)

    INPUTS:
    A     : xf_All struct

    OUTPUTS:
    Mesh    : xf_Mesh pointer
    Geom    : xf_Geom pointer
    DataSet : xf_DataSet pointer
    Param   : xf_Param pointer
    EqnSet  : xf_EqnSet pointer
*/


#endif
