#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_All.h>
#include <xf_Basis.h>


// Function to evaluate one GenArray on a basis.
PyObject*
px_InterpVector2D(PyObject *self, PyObject *args)
{
	xf_All *All;
	xf_VectorGroup *VG;
	xf_Vector *V;
	xf_GenArray *G;
	real *u;
	enum xfe_BasisType QBasis, UBasis;
	int *T;
	int i, j, ierr, egrp, nEGrp, nElem;
	int QOrder, UOrder, POrder, pPOrder;
	int nx, nT;
	int nn, nq, dim;
	
	// Parse the inputs.
	if (!PyArg_ParseTuple(args, "nn", &All, &VG))
		return NULL;
	
	// Get the vector (temporary...)
	V = VG->Vector[0];
	// Number of element groups in Mesh
	nEGrp = All->Mesh->nElemGroup;
	// That must equal number of arrays in vector
	if (nEGrp != V->nArray) {
		ierr = xf_Error(xf_INCOMPATIBLE);
		return NULL;
	}
	// Check for variable orders.
	if (V->vOrder != NULL) {
		ierr = xf_Error(xf_NOT_SUPPORTED);
		PyErr_SetString(PyExc_RuntimeError, "Variable orders not supported.");
		return NULL;
	}
	
	// Initialize counts
	nx = 0;
	nT = 0;
	// Get the total number of triangles and points.
	for (egrp=0; egrp<nEGrp; egrp++) {
		// Geometry order and basis
		QOrder = All->Mesh->ElemGroup[egrp].QOrder;
		// Approximatior order
		UOrder = V->Order[egrp];
		// Order to be used.
		POrder = (QOrder<UOrder) ? UOrder : QOrder;
		// Extract the basis.
		UBasis = V->Basis[egrp];
		// Number of elements in this group
		nElem = All->Mesh->ElemGroup[egrp].nElem;
		// Number of nodes per element
		ierr = xf_Error(xf_Order2nNode(UBasis, POrder, &nn));
		if (ierr != xf_OK) return NULL;
		// Add to number of nodes.
		nx += nElem*nn;
		// Determine nTri
		switch (UBasis) {
		case xfe_TriLagrange:
			nT += nElem*(POrder*(POrder+1) + POrder*(POrder-1))/2;
			break;
		case xfe_TriHierarch:
			nT += nElem*(POrder*(POrder+1) + POrder*(POrder-1))/2;
			break;
		case xfe_QuadLagrange:
			nT += 2*nElem*POrder*POrder;
			break;
		default:
			ierr = xf_Error(xf_NOT_SUPPORTED);
			return NULL;
			break;
		}
	} // i, GenArray (or ElementGroup) index
	
	
	
	// Output
	return Py_BuildValue("ii", nx, nT);
}
