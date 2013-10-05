#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xf_AllStruct.h>
#include <xf_ResidualStruct.h>
#include <xf_All.h>
#include <xf_Basis.h>


#define TRINODE 3

// Function to evaluate one GenArray on a basis.
PyObject*
px_InterpVector(PyObject *self, PyObject *args)
{
	xf_All *All;
	xf_ElemGroup EG;
	xf_VectorGroup *UG;
	xf_Vector *U;
	xf_GenArray *G;
	xf_BasisData *PhiQ, *PhiU;
	PyObject *PyX, *PyU, *PyT, *PyXLim;
	real **X=NULL, **u=NULL;
	real *xn, *xyN, *xyG;
	real *EU, *u0;
	real *xmin, *xmax, *xmin_i, *xmax_i;
	npy_intp dims[2];
	enum xfe_BasisType QBasis, UBasis, pUBasis, LagBasis;
	enum xfe_Bool qLim;
	enum xfe_ShapeType QShape;
	int **T;
	int *T0;
	int ierr, k, i, d, egrp, nEGrp, elem, nElem;
	int QOrder, UOrder, POrder, pPOrder;
	int nx, nT, ix, iT;
	int *igN;
	int nn, nnq, nnu, nnp, nq, nt, dim, sr, nnmax, nnpmax;
	int nnode, nsplit, nbound;
	
	// Parse the inputs.
	if (!PyArg_ParseTuple(args, "nnO", &All, &UG, &PyXLim))
		return NULL;
	// Get the element-interior state.
	ierr = xf_Error(xf_GetVectorFromGroup(UG, xfe_VectorRoleElemState, &U));
	if (ierr != xf_OK) return NULL;
	// Number of element groups in Mesh
	nEGrp = All->Mesh->nElemGroup;
	// Number of states
	sr = All->EqnSet->StateRank;
	// Dimension
	dim = All->Mesh->Dim;
	// That must equal number of arrays in vector
	if (nEGrp != U->nArray) {
		ierr = xf_Error(xf_INCOMPATIBLE);
		return NULL;
	}
	// Check for variable orders.
	if (U->vOrder != NULL) {
		ierr = xf_Error(xf_NOT_SUPPORTED);
		PyErr_SetString(PyExc_RuntimeError, "Variable orders not supported.");
		return NULL;
	}
	
	// Check the [xmin, xmax, ymin, ymax(, zmin, zmax)] list.
	if (!PyList_Check(PyXLim)) {
		PyErr_SetString(PyExc_TypeError, "Input limits must be a list");
		return NULL;
	}
	// Check the dimension.
	if ((int) PyList_Size(PyXLim) != 2*dim) {
		PyErr_SetString(PyExc_RuntimeError, "Limits must have 2*dim entries");
		return NULL;
	}
	// Allocate vectors for window min and max coordinates.
	ierr = xf_Error(xf_Alloc((void **) &xmin, dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	ierr = xf_Error(xf_Alloc((void **) &xmax, dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Allocate vectors for min and max coordinates of each element.
	ierr = xf_Error(xf_Alloc((void **) &xmin_i, dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	ierr = xf_Error(xf_Alloc((void **) &xmax_i, dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Loop through dimensions to set window bounds.
	for (d=0; d<dim; d++) {
		xmin[d] = (real) PyFloat_AsDouble(PyList_GetItem(PyXLim, 2*d));
		xmax[d] = (real) PyFloat_AsDouble(PyList_GetItem(PyXLim, 2*d+1));
	}
	
	// Initialize counts
	nx = 0; nnmax = 0; nnpmax = 0;
	nT = 0;
	// Get the total number of triangles and points.
	for (egrp=0; egrp<nEGrp; egrp++) {
		// Geometry order and basis
		QOrder = All->Mesh->ElemGroup[egrp].QOrder;
		QBasis = All->Mesh->ElemGroup[egrp].QBasis;
		// Previous order and basis
		pPOrder = -1;
		pUBasis = -1;
		// Number of elements in group
		nElem = All->Mesh->ElemGroup[egrp].nElem;
		
		// Loop through elements
		for (elem=0; elem<nElem; elem++) {
			// Get interpolation order and basis.
			xf_InterpOrderBasis(U, egrp, elem, &UOrder, &UBasis);
			// Order to be used.
			POrder = (QOrder<UOrder) ? UOrder : QOrder;
			// Redetermine count if necessary.
			if ((POrder != pPOrder) || (UBasis != pUBasis)) {
				// Remember the most recent values.
				pPOrder = POrder;
				pUBasis = UBasis;
				
				// Get the shape type (tris or tets, but just to be safe).
				ierr = xf_Error(xf_Basis2Shape(QBasis, &QShape));
				if (ierr != xf_OK) return NULL;
	
				// Set pointers.
				xn = NULL;
				T0 = NULL;
				// Get the element subdivision.
				ierr = xf_Error(xf_GetRefineCoords(QShape, POrder, &nnp, &xn,
					&nt, &T0, NULL, NULL));
				if (ierr != xf_OK) return NULL;

				// Determine nn = # unknowns for elements in this group
				ierr = xf_Error(xf_Order2nNode(UBasis, POrder, &nn));
				if (ierr != xf_OK) return NULL;
			}
			// Add to the number of elements.
			nx += nn;
			// Add to the number of triangles.
			nT += nt;
			// Update max number of nodes.
			nnmax = max(nnmax, nn);
			nnpmax = max(nnpmax, nnp);
			
		} // elem, element index 
	} // egrp, GenArray (or ElementGroup) index
	
	// Allocate coordinates.
	ierr = xf_Error(xf_Alloc2((void ***) &X, nx, dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Allocate states.
	ierr = xf_Error(xf_Alloc2((void ***) &u, nx, sr, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Allocate triangles
	ierr = xf_Error(xf_Alloc2((void ***) &T, nT, TRINODE, sizeof(int)));
	if (ierr != xf_OK) return NULL;
	// Allocate coordinates.
	ierr = xf_Error(xf_Alloc((void **) &xyN, nnmax*dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Allocate global coordinates at subdivision nodes
	ierr = xf_Error(xf_Alloc((void **) &xyG, nnpmax*dim, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	// Allocate states at subdivision nodes
	ierr = xf_Error(xf_Alloc((void **) &u0, nnpmax*sr, sizeof(real)));
	if (ierr != xf_OK) return NULL;
	
	// Initialize indices
	ix = 0;
	iT = 0;
	// Initialize data.
	xn = NULL;
	T0 = NULL;
	PhiQ = NULL;
	PhiU = NULL;
	// Get the total number of triangles and points.
	for (egrp=0; egrp<nEGrp; egrp++) {
		// Element group
		EG = All->Mesh->ElemGroup[egrp];
		// Geometry order and basis
		QOrder = EG.QOrder;
		// Previous order and basis
		pPOrder = -1;
		pUBasis = -1;
		// Number of elements in group
		nElem = EG.nElem;
		
		// Loop through elements
		for (elem=0; elem<nElem; elem++) {
			// Get interpolation order and basis.
			xf_InterpOrderBasis(U, egrp, elem, &UOrder, &UBasis);
			// Order to be used.
			POrder = (QOrder<UOrder) ? UOrder : QOrder;
			
			// Pull off the coefficients of the solution
			EU = U->GenArray[egrp].rValue[elem];
			
			// Redetermine count if necessary.
			if ((POrder != pPOrder) || (UBasis != pUBasis)) {
				// Remember the most recent values.
				pPOrder = POrder;
				pUBasis = UBasis;
				
				// Number of geometry nodes
				ierr = xf_Error(xf_Order2nNode(QBasis, QOrder, &nnq));
				if (ierr != xf_OK) return NULL;
				
				// Determine nn = # unknowns for elements in this group.
				ierr = xf_Error(xf_Order2nNode(UBasis, UOrder, &nnu));
				if (ierr != xf_OK) return NULL;
				
				// Get the shape type (tris or tets, but just to be safe).
				ierr = xf_Error(xf_Basis2Shape(QBasis, &QShape));
				if (ierr != xf_OK) return NULL;
	
				// Set pointers.
				xn = NULL;
				T0 = NULL;
				// Get the element subdivision.
				ierr = xf_Error(xf_GetRefineCoords(QShape, POrder, &nnp, &xn,
					&nt, &T0, NULL, NULL));
				if (ierr != xf_OK) return NULL;
				
				// Compute basis functions at Lagrange nodes.
				ierr = xf_Error(xf_EvalBasis(UBasis, UOrder, xfe_False, nnp, xn,
					xfb_Phi, &PhiU));
				if (ierr != xf_OK) return NULL;
				
				// Compute basis functions at Lagrange nodes.
				ierr = xf_Error(xf_EvalBasis(QBasis, QOrder, xfe_False, nnp, xn,
					xfb_Phi, &PhiQ));
				if (ierr != xf_OK) return NULL;
			}
			
			// Geometry nodes
			igN = EG.Node[elem];
			// Reset element min/max coordinates.
			for (d=0; d<dim; d++) {
				xmin_i[d] = xmax[d]+1.0;
				xmax_i[d] = xmin[d]-1.0;
			}
			// Loop through nodes.
			for (i=0; i<nnq; i++) {	
				// Loop through dimensions.
				for (d=0; d<dim; d++) {
					// Pull of node coordinates.
					xyN[i*dim+d] = All->Mesh->Coord[igN[i]][d];
					// Update element min/max coords
					xmin_i[d] = min(xyN[i*dim+d], xmin_i[d]);
					xmax_i[d] = max(xyN[i*dim+d], xmax_i[d]);
				}
			}
			// Check if the element has a vertex in the range.
			qLim = xfe_True;
			for (d=0; d<dim; d++) {
				qLim = qLim && xmax_i[d]>=xmin[d];
				qLim = qLim && xmin_i[d]<=xmax[d];
			}
			// If not in the window, skip the element.
			if (!qLim) continue;
			
			
			// Calculate global coordinates of subdivision nodes.
			xf_MxM_Set(PhiQ->Phi, xyN, nnp, nnq, dim, xyG);
			// Interpolate the state onto those nodes.
			xf_MxM_Set(PhiU->Phi, EU, nnp, nnu, sr, u0);
			
			// Loop through subdivision nodes
			for (i=0; i<nnp; i++) {
				for (d=0; d<dim; d++) {
					// Coordinates
					X[ix+i][d] = xyG[i*dim+d];
				}
				for (k=0; k<sr; k++) {
					// States
					u[ix+i][k] = u0[i*sr+k];
				}
			}
			
			// Loop through the triangles.
			for (i=0; i<nt; i++) {
				for (k=0; k<TRINODE; k++) {
					T[iT+i][k] = T0[i*TRINODE+k]+ix;
				}
			}
			
			// Add to the number of elements.
			ix += nnp;
			// Add to the number of triangles.
			iT += nt;
		} // elem, element index 
	} // i, GenArray (or ElementGroup) index
	
	
	// Number of nodes: output for Python
	dims[0] = ix;
	// Number of coordinates
	dims[1] = dim;
	// Build the coordinate array.
	PyX = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, *X);
	
	// Number of states (from EqnSet)
	dims[1] = sr;
	// Build the state array.
	PyU = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, *u);
	
	// Number of triangles
	dims[0] = iT;
	// Number of nodes per shape
	dims[1] = TRINODE;
	// Build the triangulation.
	PyT = PyArray_SimpleNewFromData(2, dims, NPY_INT, *T);
	
	// Output
	return Py_BuildValue("OOO", PyX, PyU, PyT);
}


// Function to wrap xf_GetRefineCoords
PyObject *
px_GetRefineCoords(PyObject *self, PyObject *args)
{
	enum xfe_ShapeType Shape;
	int ierr, p, dim;
	int nnode, nsplit, nbound;
	int *vsplit, *vbound;
	real *coord;
	npy_intp dims[2];
	PyObject *X, *py_vsplit, *py_vbound;
	
	// Process arguments.
	if (!PyArg_ParseTuple(args, "ii", &Shape, &p))
		return NULL;
	
	// Number of dimensions
	ierr = xf_Error(xf_Shape2Dim(Shape, &dim));
	if (ierr != xf_OK) return NULL;
	
	// Initialize pointers.
	coord = NULL;
	vsplit = NULL;
	vbound = NULL;
	// Wrap xf_GetRefineCoords.
	ierr = xf_Error(xf_GetRefineCoords(Shape, p, &nnode,
		&coord, &nsplit, &vsplit, &nbound, &vbound));
	if (ierr != xf_OK) return NULL;
	
	// Create NumPy arrays.
	// Number of points
	dims[0] = nnode;
	dims[1] = dim;
	// Create coordinate vector
	X = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, coord);
	
	// Number of subelements
	dims[0] = nsplit;
	dims[1] = dim+1;
	// Create subelement vector
	py_vsplit = PyArray_SimpleNewFromData(2, dims, NPY_INT, vsplit);
	
	// Number of subelement boundary edges/faces
	dims[0] = nbound;
	// Create subelement boundary vector
	py_vbound = PyArray_SimpleNewFromData(1, dims, NPY_INT, vbound);
	
	// Output
	return Py_BuildValue("OOO", X, py_vsplit, py_vbound);
}



