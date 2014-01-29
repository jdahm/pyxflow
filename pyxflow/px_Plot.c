#include <Python.h>

#include "px_NumPy.h"

#include "xf_AllStruct.h"
#include "xf_ResidualStruct.h"
#include "xf_All.h"
#include "xf_Mesh.h"
#include "xf_Basis.h"
#include "xf_Memory.h"
#include "xf_MeshTools.h"
#include "xf_Math.h"
#include "xf_Residual.h"
#include "xf_EqnSetHook.h"
#include "xf_Data.h"

#define DIMP 2
#define TRINN 3

/* General plotting supporting functions and objects */

typedef struct {
    int nnode; // number of nodes in refinement
    real *xref; // reference coordinates of nodes
    int nselem; // number of subelements
    int *selem; // list of subelement nodes (unrolled)
    int nsbound; // number of subelement boundary edges/faces
    int *sbound; // list of subelement boundary edges/faces

    real *xglob; // global coordinates of nodes

    enum xfe_Bool PointsChanged; // True if points have changed
    enum xfe_ShapeType Shape; // shape
    enum xfe_BasisType Basis; // basis
    int Order; // basis order
    int Face; // local face number (if subdividing faces)
    xf_BasisData *PhiData; // basis data
} SubData;

static int
InitSubData(SubData *SD)
{
    SD->nnode = 0;
    SD->xref = NULL;
    SD->nselem = 0;
    SD->selem = NULL;
    SD->nsbound = 0;
    SD->sbound = NULL;
    SD->xglob = NULL;

    SD->PointsChanged = xfe_True;
    SD->Shape = xfe_ShapeLast;
    SD->Basis = xfe_BasisLast;
    SD->Order = -1;
    SD->Face = -1;
    SD->PhiData = NULL;

    return xf_OK;
}

static int
DestroySubData(SubData *SD)
{
    int ierr;

    xf_Release((void *) SD->xref );
    xf_Release((void *) SD->selem );
    xf_Release((void *) SD->sbound );
    xf_Release((void *) SD->xglob );

    ierr = xf_Error(xf_DestroyBasisData(SD->PhiData, xfe_False));

    if (ierr != xf_OK) return ierr;

    return xf_OK;
}

static int
FindElemSubData(xf_Mesh *Mesh, int egrp, int elem, const int *pOrder, SubData *SD)
{
    int ierr, dim, pnn, Order;
    enum xfe_BasisType QBasis, QOrder;
    enum xfe_ShapeType Shape;

    dim = Mesh->Dim;

    QOrder = Mesh->ElemGroup[egrp].QOrder;
    QBasis = Mesh->ElemGroup[egrp].QBasis;

    Order = ((pOrder != NULL) ? (*pOrder) : 2*QOrder+1);

    ierr = xf_Error(xf_Basis2Shape(QBasis, &Shape));

    if (ierr != xf_OK) return ierr;

    SD->PointsChanged = xfe_False;

    // Do we need to recalculate?
    if ((Shape != SD->Shape) || (Order != SD->Order)) {
        SD->PointsChanged = xfe_True;
        SD->Shape = Shape;
        SD->Order = Order;

        pnn = SD->nnode;

        // Get the element subdivision
        ierr = xf_Error(xf_GetRefineCoords(Shape, Order, &SD->nnode, &SD->xref,
                                           &SD->nselem, &SD->selem, &SD->nsbound, &SD->sbound));

        if (ierr != xf_OK) return ierr;

        // Reallocate xref if needed
        if (pnn != SD->nnode) {
            ierr = xf_Error(xf_ReAlloc((void **) &SD->xglob, dim * SD->nnode, sizeof(real)));

            if (ierr != xf_OK) return ierr;
        }
    }

    ierr = xf_Error(xf_Ref2GlobElem(Mesh, egrp, elem, &SD->PhiData, SD->PointsChanged,
                                    SD->nnode, SD->xref, SD->xglob));

    if (ierr != xf_OK) return ierr;

    return xf_OK;
}

static int
FindFaceSubData(xf_Mesh *Mesh, int ibfgrp, int ibface, const int *pOrder, SubData *SD)
{
    int ierr, dim, pnn, Order, egrp, elem, face;
    int dbfgrp, dbface;
    enum xfe_BasisType QBasis, QOrder;
    enum xfe_ShapeType Shape, FShape;

    dim = Mesh->Dim;

    // Get data from left element
    xf_FaceElements(Mesh, ibfgrp, ibface, &egrp, &elem, &face, NULL, NULL, NULL);

    QOrder = Mesh->ElemGroup[egrp].QOrder;
    QBasis = Mesh->ElemGroup[egrp].QBasis;

    Order = ((pOrder != NULL) ? (*pOrder) : 2*QOrder+1);

    ierr = xf_Error(xf_Basis2Shape(QBasis, &Shape));

    if (ierr != xf_OK) return ierr;

    SD->PointsChanged = xfe_False;

    // Do we need to recalculate?
    if ((Shape != SD->Shape) || (Order != SD->Order) || (face != SD->Face)) {
        SD->PointsChanged = xfe_True;
        SD->Shape = Shape;
        SD->Order = Order;
        SD->Face = face;

        pnn = SD->nnode;

        // get shape on face
        ierr = xf_Error(xf_FaceShape(Shape, face, &FShape));

        if (ierr != xf_OK) return ierr;
  
        // refinement coords on face
        ierr = xf_Error(xf_GetRefineCoords(FShape, Order, &SD->nnode, &SD->xref, &SD->nselem,
                    &SD->selem, &SD->nsbound, &SD->sbound));

        if (ierr != xf_OK) return ierr;

        // Reallocate xref if needed
        if (pnn != SD->nnode) {
            ierr = xf_Error(xf_ReAlloc((void **) &SD->xglob, dim * SD->nnode, sizeof(real)));

            if (ierr != xf_OK) return ierr;
        }
    }

    // Convert to data numbering
    //xf_FaceMeshGroup2DataGroup(Mesh, ibfgrp, ibface, &dbfgrp, &dbface);

    ierr = xf_Error(xf_Ref2GlobFace(Mesh, ibfgrp+1, ibface, &SD->PhiData,
                SD->nnode, SD->xref, SD->xglob));

    if (ierr != xf_OK) return ierr;

    return xf_OK;
}


static enum xfe_Bool
PointInsideBoundingBox(int dim, const real *x, const real *xmin, const real *xmax, real buffer)
{
    int d;
    enum xfe_Bool Inside = xfe_True;
    real min, max;

    for (d = 0; (d < dim) && Inside; d++) {
        min = xmin[d] - (xmax[d] - xmin[d]) * buffer;
        max = xmax[d] + (xmax[d] - xmin[d]) * buffer;

        if ( (x[d] < min) || (x[d] > max) ) Inside = xfe_False;
    }

    return Inside;
}

static int
ElemInsideBoundingBox(const xf_Mesh *Mesh, int egrp, int elem, const real *xmin, const real *xmax,
                      real buffer, enum xfe_Bool *Inside)
{
    /*  Simply check the nodes on the element for now In the future,
        maybe should check the global positions of nodes on faces, since
        nodes could be completely interior to the element. */

    int ierr, dim, i, d, inode, Order, nn;
    enum xfe_BasisType Basis;
    real xglob[xf_MAXDIM];

    dim = Mesh->Dim;

    Order = Mesh->ElemGroup[egrp].QOrder;
    Basis = Mesh->ElemGroup[egrp].QBasis;

    ierr = xf_Error(xf_Order2nNode(Basis, Order, &nn));

    if (ierr != xf_OK) return ierr;

    (*Inside) = xfe_False;

    for (i = 0; (i < nn) && (!(*Inside)); i++) {
        inode = Mesh->ElemGroup[egrp].Node[elem][i];

        for (d = 0; d < dim; d++)
            xglob[d] = Mesh->Coord[inode][d];

        (*Inside) = PointInsideBoundingBox(dim, xglob, xmin, xmax, buffer);
    }

    return xf_OK;
}

static int
UnpackRealList(PyObject *PyL, int n, real *l, enum xfe_Bool fit)
{
    int i;

    // Check type
    if (!PyList_Check(PyL)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list");
        return xf_MEMORY_ERROR;
    }

    // Check dimension
    if (fit && ((int) PyList_Size(PyL) != n)) {
        PyErr_SetString(PyExc_RuntimeError, "Input has incorrect dimensions");
        return xf_MEMORY_ERROR;
    }

    // Unpack
    for (i = 0; i < n; i++)
        l[i] = (real) PyFloat_AsDouble(PyList_GetItem(PyL, i));

    return xf_OK;
}

/* Mesh plotting */

typedef struct {
    real *x; // global coordinates of nodes on face
    int *nn; // number of nodes
    int xsize; // size of xglob allocated
    int nnsize; // size of nn allocated
    int nface; // number of faces
} MeshPlotData;

static int
InitMeshPlotData(MeshPlotData *MPD)
{
    MPD->x  = NULL;
    MPD->nn = NULL;
    MPD->xsize = 0;
    MPD->nnsize = 0;
    MPD->nface = 0;
    return xf_OK;
}

static int
DestroyMeshPlotData(MeshPlotData *MPD)
{
    xf_Release((void *) MPD->x );
    xf_Release((void *) MPD->nn);

    return xf_OK;
}


static int
MeshPlotData_1D(xf_Mesh *Mesh, int egrp, int elem, SubData *FSD, MeshPlotData *MPD)
{
    int ierr, nface, face, nn, i, f;
    real xref[xf_MAXDIM], xglob[xf_MAXDIM];
    enum xfe_Bool OnLeft;
    enum xfe_FaceType FaceType;
    xf_IFace IFace;
    xf_Face Face;

    if ((nface = Mesh->ElemGroup[egrp].nFace[elem]) != 2) return xf_CODE_LOGIC_ERROR;

    // Reallocate MPD->nn if necessary
    if (MPD->nnsize < nface) {
        MPD->nnsize = nface;
        ierr = xf_Error(xf_ReAlloc((void **)&MPD->nn, MPD->nnsize, sizeof(int)));

        if (ierr != xf_OK) return ierr;
    }

    // in 1D, the node is always at reference coordinate xi=0. on the face
    xref[0] = 0.;

    for (face = 0, i = 0, f = 0; face < nface; face++) {
        Face = Mesh->ElemGroup[egrp].Face[elem][face];
        xf_FaceGroupInfo(Mesh, Face.Group, &FaceType, NULL, NULL);

        if (FaceType == xfe_FaceInterior) {
            IFace = Mesh->IFace[Face.Number];

            ierr = xf_Error(xf_IsElemOnLeft(IFace, egrp, elem, &OnLeft));

            if (ierr != xf_OK) return ierr;
        } else OnLeft = xfe_True;

        if (!OnLeft) continue;

        // always only one node on the face
        nn = 1;

        // Reallocate MPD->x if necessary
        if (MPD->xsize < DIMP * (i + nn)) {
            MPD->xsize = DIMP * (i + nn);
            ierr = xf_Error(xf_ReAlloc((void **)&MPD->x, MPD->xsize, sizeof(real)));

            if (ierr != xf_OK) return ierr;
        }

        ierr = xf_Error(xf_Ref2GlobFace(Mesh, Face.Group, Face.Number, &FSD->PhiData, nn, xref, xglob));

        if (ierr != xf_OK) return ierr;

        // in 1D, y-position in plot is set to 0.0
        MPD->x[i * DIMP + 0] = xglob[0];
        MPD->x[i * DIMP + 1] = 0.0;

        MPD->nn[f] = 1;

        f++;
        i += nn;
    } // face

    // store number of faces in view
    MPD->nface = f;

    return xf_OK;
}

static int
MeshPlotData_2D(xf_Mesh *Mesh, int egrp, int elem, int *pOrder, SubData *FSD, MeshPlotData *MPD)
{
    int ierr, nface, face, nn, i, f, ibfgrp, ibface;
    enum xfe_Bool OnLeft;
    enum xfe_FaceType FaceType;
    xf_IFace IFace;
    xf_Face Face;

    nface = Mesh->ElemGroup[egrp].nFace[elem];

    // Reallocate MPD->nn if necessary
    if (MPD->nnsize < nface) {
        MPD->nnsize = nface;
        ierr = xf_Error(xf_ReAlloc((void **)&MPD->nn, MPD->nnsize, sizeof(int)));

        if (ierr != xf_OK) return ierr;
    }

    for (face = 0, i = 0, f = 0; face < nface; face++) {
        Face = Mesh->ElemGroup[egrp].Face[elem][face];
        xf_FaceGroupInfo(Mesh, Face.Group, &FaceType, &ibfgrp, NULL);

        if (FaceType == xfe_FaceInterior) {
            IFace = Mesh->IFace[Face.Number];

            ierr = xf_Error(xf_IsElemOnLeft(IFace, egrp, elem, &OnLeft));

            if (ierr != xf_OK) return ierr;
        } else OnLeft = xfe_True;

        if (!OnLeft) continue;

        ierr = xf_Error(FindFaceSubData(Mesh, ibfgrp, Face.Number, pOrder, FSD));

        if (ierr != xf_OK) return ierr;

        nn = FSD->nnode;

        // Reallocate MPD->x if necessary (over-allocating for speed)
        if (MPD->xsize < DIMP * (i + nn)) {
            MPD->xsize = 2 * DIMP * (i + nn);
            ierr = xf_Error(xf_ReAlloc((void **)&MPD->x, MPD->xsize, sizeof(real)));

            if (ierr != xf_OK) return ierr;
        }

        // copy global coordinates
        xf_V_Add(FSD->xglob, DIMP * nn, xfe_Set, MPD->x + DIMP * i);

        MPD->nn[f] = nn;

        f++;
        i += nn;
    } // face

    // store number of faces in view
    MPD->nface = f;

    return xf_OK;
}

static int
MeshPlotData_3D(xf_Mesh *Mesh, int egrp, int elem, int *pOrder, SubData *ESD, SubData *FSD, MeshPlotData *MPD)
{
    return xf_NOT_SUPPORTED;
}

/* Scalar plotting */

typedef struct {
    int egrp, elem; // element index
    int *IParam; // integer parameter array for EqnSet
    real *RParam; // real-valued parameter array for EqnSet
    int nAux; // number of auxiliary vectors
    xf_Vector **VAux; // pointers to auxiliary vectors
    real *U, *gU; // storage for state and gradient at plotting nodes
    int Usize, gUsize; // size of storage at plotting nodes
    real *s; // storage for scalar at plotting nodes
    int ssize; // size of storage for scalar at plotting nodes
    xf_BasisTable *PhiTable; // table to avoid recalculation of basis functions
    xf_BasisData *BD; // basis data for interpolating scalar
    xf_JacobianData *JD; // Jacobian data for calculating gradients
} ScalarPlotData;

static int
InitScalarPlotData(xf_EqnSet *EqnSet, ScalarPlotData *SPD)
{
    int ierr;

    SPD->egrp = SPD->elem = 0;

    // communicate with EqnSet to fill real and integer parameters
    ierr = xf_Error(xf_RetrieveFcnParams(NULL, EqnSet, &SPD->IParam, &SPD->RParam,
                                         &SPD->nAux, &SPD->VAux));

    if (ierr != xf_OK) return ierr;

    SPD->U = SPD->gU = NULL;
    SPD->Usize = SPD->gUsize = 0;
    SPD->s = NULL;
    SPD->ssize = 0;

    ierr = xf_Error(xf_CreateBasisTable(&SPD->PhiTable));

    if (ierr != xf_OK) return ierr;

    SPD->BD = NULL;
    SPD->JD = NULL;

    return xf_OK;
}

static int
DestroyScalarPlotData(ScalarPlotData *SPD)
{
    int ierr;

    xf_Release((void *) SPD->IParam);
    xf_Release((void *) SPD->RParam);
    xf_Release((void *) SPD->VAux);
    xf_Release((void *) SPD->U );
    xf_Release((void *) SPD->gU);

    ierr = xf_Error(xf_DestroyBasisData(SPD->BD, xfe_False));

    if (ierr != xf_OK) return ierr;

    ierr = xf_Error(xf_DestroyBasisTable(SPD->PhiTable));

    if (ierr != xf_OK) return ierr;

    ierr = xf_Error(xf_DestroyJacobianData(SPD->JD));

    if (ierr != xf_OK) return ierr;

    return xf_OK;
}

static int
ScalarValues(xf_Vector *U, xf_Mesh *Mesh, xf_EqnSet *EqnSet, int egrp, int elem, char *Name, SubData *ESD, ScalarPlotData *SPD)
{
    int ierr, i, d, dim, sr, nn, nq, Order;
    enum xfe_Bool Interpolated;
    enum xfe_BasisType Basis;
    real *EU;

    dim = Mesh->Dim;
    sr = U->StateRank;

    nq = ESD->nnode;

    EU = U->GenArray[egrp].rValue[elem];

    // Reallocate if necessary
    if (SPD->Usize < sr * nq) {
        SPD->Usize = sr * nq;
        ierr = xf_Error(xf_ReAlloc((void **)&SPD->U, SPD->Usize, sizeof(real)));

        if (ierr != xf_OK) return ierr;
    }

    if (SPD->gUsize < dim * sr * nq) {
        SPD->gUsize = dim * sr * nq;
        ierr = xf_Error(xf_ReAlloc((void **)&SPD->gU, SPD->gUsize, sizeof(real)));

        if (ierr != xf_OK) return ierr;
    }

    if (SPD->ssize < nq) {
        SPD->ssize = nq;
        ierr = xf_Error(xf_ReAlloc((void **)&SPD->s, SPD->ssize, sizeof(real)));

        if (ierr != xf_OK) return ierr;
    }

    Interpolated = ((U->Basis != NULL) && (U->Order != NULL));

    if (!Interpolated && (Name != NULL)) return xf_NOT_SUPPORTED;

    if (Interpolated) {
        xf_InterpOrderBasis(U, egrp, elem, &Order, &Basis);

        ierr = xf_Error(xf_Order2nNode(Basis, Order, &nn));

        if (ierr != xf_OK) return ierr;

        // Evaluate the basis functions for the vector
        ierr = xf_Error(xf_EvalBasisUsingTable(Basis, Order, ESD->PointsChanged, ESD->nnode, ESD->xref,
                                               xfb_Phi | xfb_GPhi | xfb_gPhi, SPD->PhiTable, &SPD->BD));

        if (ierr != xf_OK) return ierr;

        // values
        xf_MxM_Set(SPD->BD->Phi, EU, nq, nn, sr, SPD->U);
        
        if (Name != NULL) {
            // element Jacobian det and inv at points (only 1 if J is const)
            ierr = xf_Error(xf_ElemJacobian(Mesh, egrp, elem, ESD->nnode, ESD->xref, xfb_detJ | xfb_iJ,
                                            ESD->PointsChanged, &SPD->JD));

            if (ierr != xf_OK) return ierr;

            ierr = xf_Error(xf_EvalPhysicalGrad(SPD->BD, SPD->JD));

            if (ierr != xf_OK) return ierr;

            // gradients
            for (d = 0; d < dim; d++)
                xf_MxM_Set(SPD->BD->gPhi + nn * nq * d, EU, nq, nn, sr, SPD->gU + nq * sr * d);

            ierr = xf_Error(xf_EqnSetScalar(EqnSet, Name, SPD->IParam, SPD->RParam, nq,
                                            SPD->U, SPD->gU, SPD->s, NULL, NULL, NULL));

            if (ierr != xf_OK) return ierr;
        } else for (i = 0; i < nq; i++) SPD->s[i] = SPD->U[i * sr];
    } else for (i = 0; i < nq; i++) SPD->s[i] = EU[0];


    return xf_OK;
}


static int
ScalarPlotData_3D(xf_Vector *U, xf_Mesh *Mesh, xf_EqnSet *EqnSet, int egrp, int elem, SubData *ESD)
{
    return xf_NOT_SUPPORTED;
}


PyObject*
px_MeshPlotData(PyObject *self, PyObject *args)
{
    int ierr, dim, i, nn, nntotal, egrp, elem;
    int Order, *pOrder;
    npy_intp pydim[3];
    enum xfe_Bool Inside;
    real xmin[xf_MAXDIM], xmax[xf_MAXDIM];
    real *x, *y;
    int psize, np;
    int *c;
    int csize, nc;
    SubData ESD, FSD;
    MeshPlotData MPD;
    PyObject *py_x, *py_y, *py_c, *py_min, *py_max, *py_order;
    xf_Mesh *Mesh;

    // Parse the inputs.
    if (!PyArg_ParseTuple(args, "nOOO", &Mesh, &py_min, &py_max, &py_order))
        return NULL;

    dim = Mesh->Dim;

    ierr = xf_Error(UnpackRealList(py_min, dim, xmin, xfe_True));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(UnpackRealList(py_max, dim, xmax, xfe_True));

    if (ierr != xf_OK) return NULL;

    if (PyInt_Check(py_order)){
        Order = (int) PyInt_AsLong(py_order);
        pOrder = &Order;
    }
    else{
        pOrder= NULL;
    }

    // Init the face and element data
    ierr = xf_Error(InitMeshPlotData(&MPD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(InitSubData(&ESD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(InitSubData(&FSD));

    if (ierr != xf_OK) return NULL;

    x = y = NULL;
    psize = 0;
    np = 0;
    c = NULL;
    nc = 0;
    csize = 0;
    nn = 0;

    for (egrp = 0; egrp < Mesh->nElemGroup; egrp++) {
        for (elem = 0; elem < Mesh->ElemGroup[egrp].nElem; elem++) {
            // check if element is inside window
            ierr = xf_Error(ElemInsideBoundingBox(Mesh, egrp, elem, xmin, xmax, 0.5, &Inside));

            if (ierr != xf_OK) return NULL;

            if (!Inside) continue;

            if (dim == 1) {
                ierr = xf_Error(MeshPlotData_1D(Mesh, egrp, elem, &FSD, &MPD));

                if (ierr != xf_OK) return NULL;
            } else if (dim == 2) {
                ierr = xf_Error(MeshPlotData_2D(Mesh, egrp, elem, pOrder, &FSD, &MPD));

                if (ierr != xf_OK) return NULL;
            } else if (dim == 3) {
                ierr = xf_Error(MeshPlotData_3D(Mesh, egrp, elem, pOrder, &ESD, &FSD, &MPD));

                if (ierr != xf_OK) return NULL;
            } else return NULL;

            // Add data
            for(i = 0, nntotal = 0; i < MPD.nface; i++) nntotal += MPD.nn[i];

            // reallocate x and y if necessary
            if (psize < (np + nntotal)) {
                // larger than necessary, hopefully reducing the number of reallocs
                psize = 2 * (np + nntotal);
                ierr = xf_Error(xf_ReAlloc((void **)&x, psize, sizeof(real)));

                if (ierr != xf_OK) return NULL;

                ierr = xf_Error(xf_ReAlloc((void **)&y, psize, sizeof(real)));

                if (ierr != xf_OK) return NULL;
            }

            // node position data
            for (i = 0; i < nntotal; i++) {
                x[np + i] = MPD.x[DIMP * i];
                y[np + i] = MPD.x[DIMP * i + 1];
            }

            np += nntotal;

            // reallocate connectivity data if necessary
            if (csize < (nc + MPD.nface)) {
                // larger than necessary, hopefully reducing the number of reallocs
                csize = 2 * (nc + MPD.nface);
                ierr = xf_Error(xf_ReAlloc((void **)&c, csize, sizeof(int)));

                if (ierr != xf_OK) return NULL;
            }

            // connectivity data
            for (i = 0; i < MPD.nface; i++) {
                c[nc + i] = nn;
                nn += MPD.nn[i];
            }

            nc += MPD.nface;

        } // elem
    } // egrp

    // Trim
    ierr = xf_Error(xf_ReAlloc((void **)&x, np, sizeof(real)));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(xf_ReAlloc((void **)&y, np, sizeof(real)));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(xf_ReAlloc((void **)&c, nc, sizeof(int)));

    if (ierr != xf_OK) return NULL;

    // Convert to python arrays
    // positions
    pydim[0] = np;
    py_x = PyArray_SimpleNewFromData(1, pydim, NPY_DOUBLE, (void *)x);
    py_y = PyArray_SimpleNewFromData(1, pydim, NPY_DOUBLE, (void *)y);
    // connectivity
    pydim[0] = nc;
    py_c = PyArray_SimpleNewFromData(1, pydim, NPY_INT, (void *)c);

    // clean up
    ierr = xf_Error(DestroyMeshPlotData(&MPD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(DestroySubData(&ESD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(DestroySubData(&FSD));

    if (ierr != xf_OK) return NULL;

    return Py_BuildValue("OOO", py_x, py_y, py_c);
}

PyObject*
px_ScalarPlotData(PyObject *self, PyObject *args)
{
    int ierr, dim, egrp, elem, i;
    int QOrder, UOrder, Order;
    enum xfe_Bool FixedOrder, Inside;
    real xmin[xf_MAXDIM], xmax[xf_MAXDIM];
    npy_intp pydim[3];
    PyObject *py_c, *py_tri, *py_x, *py_y, *py_scalar, *py_min, *py_max, *py_order;
    real *x, *y, *c;
    int psize, np, *tri, ntri, trisize, csize;
    char *ScalarName;
    xf_Vector *U;
    xf_Mesh *Mesh;
    xf_EqnSet *EqnSet;
    SubData ESD;
    ScalarPlotData SPD;

    // Parse the inputs.
    if (!PyArg_ParseTuple(args, "nnnOOOO", &U, &Mesh, &EqnSet, &py_scalar, &py_min, &py_max, &py_order))
        return NULL;

    dim = Mesh->Dim;

    ierr = xf_Error(UnpackRealList(py_min, dim, xmin, xfe_True));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(UnpackRealList(py_max, dim, xmax, xfe_True));

    if (ierr != xf_OK) return NULL;

    // Call xf_EqnSetScalar or just use the first entry in the vector?
    ScalarName = PyString_Check(py_scalar) ? PyString_AsString(py_scalar) : NULL;

    // Was the requested plot order passed?
    FixedOrder = PyInt_Check(py_order);
    if (FixedOrder)
        Order = (int) PyInt_AsLong(py_order);

    ierr = xf_Error(InitScalarPlotData(EqnSet, &SPD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(InitSubData(&ESD));

    if (ierr != xf_OK) return NULL;

    x = y = NULL;
    psize = 0;
    np = 0;
    tri = NULL;
    trisize = 0;
    ntri = 0;
    c = NULL;
    csize = 0;

    for (egrp = 0; egrp < Mesh->nElemGroup; egrp++) {
        QOrder = Mesh->ElemGroup[egrp].QOrder;
        for (elem = 0; elem < Mesh->ElemGroup[egrp].nElem; elem++) {
            // check if element is inside window
            ierr = xf_Error(ElemInsideBoundingBox(Mesh, egrp, elem, xmin, xmax, 0.5, &Inside));

            if (ierr != xf_OK) return NULL;

            if (!Inside) continue;

            if ((dim >= 1) && (dim < 3)) {
                // Simple in this case
                if (!FixedOrder){
                    UOrder = xf_InterpOrder(U, egrp, elem);
                    Order = 2 * max(QOrder, UOrder) + 1;
                }
                ierr = xf_Error(FindElemSubData(Mesh, egrp, elem, &Order, &ESD));

                if (ierr != xf_OK) return NULL;
            } else if (dim == 3) {
                // Harder, need to calculate the xref along the cut plane
                ierr = xf_Error(ScalarPlotData_3D(U, Mesh, EqnSet, egrp, elem, &ESD));

                if (ierr != xf_OK) return NULL;
            } else return NULL;

            ierr = xf_Error(ScalarValues(U, Mesh, EqnSet, egrp, elem, ScalarName, &ESD, &SPD));

            if (ierr != xf_OK) return NULL;

            // reallocate x and y if necessary
            if (psize < DIMP * (np + ESD.nnode)) {
                psize = 2 * DIMP * (np + ESD.nnode);
                ierr = xf_Error(xf_ReAlloc((void **)&x, psize, sizeof(real)));

                if (ierr != xf_OK) return NULL;

                ierr = xf_Error(xf_ReAlloc((void **)&y, psize, sizeof(real)));

                if (ierr != xf_OK) return NULL;
            }

            // reallocate c if necessary
            if (csize < np + ESD.nnode) {
                csize = 2 * (np + ESD.nnode);
                ierr = xf_Error(xf_ReAlloc((void **)&c, csize, sizeof(real)));

                if (ierr != xf_OK) return NULL;
            }

            for (i = 0; i < ESD.nnode; i++) {
                // node position data
                x[np + i] = ESD.xglob[DIMP * i];
                y[np + i] = ESD.xglob[DIMP * i + 1];
                // scalar values
                c[np + i] = SPD.s[i];
            }

            // reallocate tri if necessary
            if (trisize < TRINN * (ntri + ESD.nselem)) {
                trisize = 2 * TRINN * (ntri + ESD.nselem);
                ierr = xf_Error(xf_ReAlloc((void **)&tri, trisize, sizeof(real)));

                if (ierr != xf_OK) return NULL;
            }

            // sub-triangle data
            for (i = 0; i < TRINN * ESD.nselem; i++)
                tri[TRINN * ntri + i] = np + ESD.selem[i];

            ntri += ESD.nselem;

            np += ESD.nnode;

        } // elem
    } // egrp

    // Trim
    ierr = xf_Error(xf_ReAlloc((void **)&x, np, sizeof(real)));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(xf_ReAlloc((void **)&y, np, sizeof(real)));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(xf_ReAlloc((void **)&c, np, sizeof(real)));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(xf_ReAlloc((void **)&tri, TRINN * ntri, sizeof(int)));

    if (ierr != xf_OK) return NULL;

    // Convert to python arrays
    // positions
    pydim[0] = np;
    py_x = PyArray_SimpleNewFromData(1, pydim, NPY_DOUBLE, (void *)x);
    py_y = PyArray_SimpleNewFromData(1, pydim, NPY_DOUBLE, (void *)y);
    // scalar (c)
    pydim[0] = np;
    py_c = PyArray_SimpleNewFromData(1, pydim, NPY_DOUBLE, (void *)c);
    // triangles
    pydim[0] = ntri;
    pydim[1] = TRINN;
    py_tri = PyArray_SimpleNewFromData(2, pydim, NPY_INT, (void *)tri);

    ierr = xf_Error(DestroyScalarPlotData(&SPD));

    if (ierr != xf_OK) return NULL;

    ierr = xf_Error(DestroySubData(&ESD));

    if (ierr != xf_OK) return NULL;

    return Py_BuildValue("OOOO", py_x, py_y, py_tri, py_c);
}
