#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Basis.h>
#include <xf_Geom.h>
#include <xf_GeomIO.h>
#include <xf_String.h>


/******************************************************************/
// Function to create an empty geom.
PyObject *
px_CreateGeom(PyObject *self, PyObject *args)
{
    xf_Geom *Geom = NULL;
    int ierr;

    // Create the geom.
    ierr = xf_Error(xf_CreateGeom(&Geom));
    if (ierr != xf_OK) return NULL;

    // Return the pointer.
    return Py_BuildValue("n", Geom);
}


/******************************************************************/
// Function to read geom from .geom file
PyObject *
px_ReadGeomFile(PyObject *self, PyObject *args)
{
    xf_Geom *Geom = NULL;
    char *fname;
    int ierr;

    // Parse the python inputs.
    if (!PyArg_ParseTuple(args, "s", &fname))
        return NULL;

    // Create the geom.
    ierr = xf_Error(xf_CreateGeom(&Geom));
    if (ierr != xf_OK) return NULL;

    // Read the .gri file into the xf_Mesh structure.
    ierr = xf_Error(xf_ReadGeomFile(fname, NULL, Geom));
    if (ierr != xf_OK) return NULL;

    // Return the pointer.
    return Py_BuildValue("n", Geom);
}


/******************************************************************/
// Function to write geom to .geom file
PyObject *
px_WriteGeomFile(PyObject *self, PyObject *args)
{
    xf_Geom *Geom = NULL;
    char *fname;
    int ierr;

    // Parse the Python inputs.
    if (!PyArg_ParseTuple(args, "ns", &Geom, &fname))
        return NULL;

    // Write the file.
    ierr = xf_Error(xf_WriteGeomFile(Geom, fname));
    if (ierr != xf_OK) return NULL;

    // Nothing to return.
    Py_INCREF(Py_None);
    return Py_None;
}


/******************************************************************/
// Function to extract the number of geom components
PyObject *
px_nGeomComp(PyObject *self, PyObject *args)
{
    xf_Geom *Geom = NULL;

    // Get the pointer to the xf_Mesh.
    if (!PyArg_ParseTuple(args, "n", &Geom))
        return NULL;

    // Output
    return Py_BuildValue("i", Geom->nComp);
}


/******************************************************************/
// Function to read the Geom->Comp information
PyObject *
px_GeomComp(PyObject *self, PyObject *args)
{
    xf_Geom *Geom;
    xf_GeomComp *GC;
    xf_GeomCompSpline *GCS;
    xf_GeomCompPanel *GCP;
    PyObject *D;
    int ierr, iComp, Order;
    char *Name, *BFGTitle, *Type;

    // Get the pointer to the xf_BFaceGroup
    if (!PyArg_ParseTuple(args, "ni", &Geom, &iComp))
        return NULL;

    // Check the value of iComp.
    if (iComp >= Geom->nComp) {
        PyErr_SetString(PyExc_RuntimeError, \
                        "Component index exceeds dimensions.");
        return NULL;
    }
    
    // Pointer to component
    GC = Geom->Comp + iComp;

    // Read the name.
    Name = GC->Name;
    // Convert Type to string.
    Type = xfe_GeomCompName[GC->Type];
    // Name of corresponding boundary condition
    BFGTitle = GC->BFGTitle;

    // Determine the component type
    switch(Geom->Comp[iComp].Type) {
    case xfe_GeomCompSpline:
        // Get the spline pointer.
        GCS = (xf_GeomCompSpline *) GC->Data;
        // Get the spline interpolation order.
        Order = GCS->Order;
        // Number of points
        int N = GCS->N;
        npy_intp dims[1] = {N};
        // coordinates
        PyObject *X = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, GCS->X);
        PyObject *Y = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, GCS->Y);
        // Create the object.
        D = Py_BuildValue("{sisisOsO}", \
            "Order", Order, "N", N, "X", X, "Y", Y);
        break;
    
    case xfe_GeomCompPanel:
        // Get the panel pointer.
        GCP = (xf_GeomCompPanel *) GC->Data;
        // Get the panel order.
        Order = GCP->Order;
        // Basis type.
        char *Basis;
        Basis = xfe_BasisName[GCP->Basis];
        // Number of nodes in paneling
        int nNode = GCP->nNode;
        // Number of panels
        int nPanel = GCP->nPanel;
        // Dimension
        int dim = GCP->dim;
        // Dimensions for reading the nodes.
        npy_intp dim2[2] = {nNode, dim};
        // Node coordinates
        PyObject *Coord = PyArray_SimpleNewFromData( \
            2, dim2, NPY_DOUBLE, *GCP->Coord);
        // Get the number of nodes in a panel.
        int nn;
        ierr = xf_Error(xf_Order2nNode(GCP->Basis, Order, &nn));
        if (ierr != xf_OK) return ierr;
        // Dimensions for the panel node indices.
        dim2[0] = nPanel;
        dim2[1] = nn;
        // Panel node indices (generalization of triangulation)
        PyObject *Panels = PyArray_SimpleNewFromData( \
            2, dim2, NPY_INT, *GCP->Panels);
        // Create the output object.
        D = Py_BuildValue("{sisisssisisOsO}", "Order", Order, "dim", dim, \
            "Basis", Basis, "nNode", nNode, "nPanel", nPanel, \
            "Coord", Coord, "Panels", Panels);
        break;
        
    default:
        // Return `None` for the data
        Py_INCREF(Py_None);
        D = Py_None;
        break;
    }

    // Output: (Title, nBFace, _BFace[0])
    return Py_BuildValue("sssO", Name, Type, BFGTitle, D);
}


/******************************************************************/
// Function to assign nodes to a panel
PyObject *
px_SetGeomCompPanelCoord(PyObject *self, PyObject *args)
{
    xf_Geom *Geom;
    xf_GeomComp *GC;
    xf_GeomCompPanel *GCP;
    int ierr, i, nNode, dim;
    PyObject *Py_Coord;
    real **Coord;
    
    // Get the pointer and integer inputs.
    if (!PyArg_ParseTuple(args, "niiiO", &Geom, &i, &nNode, &dim, &Py_Coord))
        return NULL;
    printf("i = %i\n", i);
    printf("nComp = %i\n", Geom->nComp);
    // Extract the panel.
    GC = Geom->Comp+i;
    printf("Label 1?\n");
    GCP = GC->Data;
    // Assign the node count.
    GCP->nNode = nNode;
    // Assign the component dimension.
    GCP->dim = dim;
    
    xf_printf("nNode: %i\n", GCP->nNode);
    xf_printf("dim: %i\n", GCP->dim);
    
    // Return `None`.
    Py_INCREF(Py_None);
    return Py_None;
}


/******************************************************************/
// Function to destroy the mesh
PyObject *
px_DestroyGeom(PyObject *self, PyObject *args)
{
    int ierr;
    xf_Geom *Geom;

    // Get the pointer.
    if (!PyArg_ParseTuple(args, "n", &Geom)) return NULL;

    // Deallocate the mesh.
    ierr = xf_Error(xf_DestroyGeom(Geom));
    if (ierr != xf_OK) return NULL;

    // Nothing to return.
    Py_INCREF(Py_None);
    return Py_None;
}

