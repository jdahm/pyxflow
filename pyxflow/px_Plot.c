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
px_InterpVector2D(PyObject *self, PyObject *args)
{
	xf_All *All;
	xf_ElemGroup EG;
	xf_VectorGroup *UG;
	xf_Vector *U;
	xf_GenArray *G;
	xf_BasisData *PhiQ, *PhiU;
	PyObject *PyX, *PyU, *PyT;
	real **X=NULL, **u=NULL;
	real *xn, *xyN, *xyG;
	real *EU, *u0;
	npy_intp dims[2];
	enum xfe_BasisType QBasis, UBasis, pUBasis, LagBasis;
	int **T;
	int *T0;
	int ierr, k, i, d, egrp, nEGrp, elem, nElem;
	int QOrder, UOrder, POrder, pPOrder;
	int nx, nT, ix, iT;
	int *igN;
	int nn, nnq, nnu, nnp, nq, nt, dim, sr, nnmax, nnpmax;
	
	// Parse the inputs.
	if (!PyArg_ParseTuple(args, "nn", &All, &UG))
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
				
				// Determine nn = # unknowns for elements in this group
				ierr = xf_Error(xf_Order2nNode(UBasis, POrder, &nn));
				if (ierr != xf_OK) return NULL;

				// Determine nn = # unknowns for elements in this group
				ierr = xf_Error(xf_Order2nNode(QBasis, POrder, &nnp));
				if (ierr != xf_OK) return NULL;
				
				// Determine number of triangles in element.
				ierr = xf_Error(px_Basis2Tri(UBasis, POrder, &nt, NULL));
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
				
				// Determine nn = # unknowns for elements in this group
				ierr = xf_Error(xf_Order2nNode(UBasis, UOrder, &nnu));
				if (ierr != xf_OK) return NULL;
				
				// Determine nn = # unknowns for elements in this group
				ierr = xf_Error(xf_Order2nNode(QBasis, POrder, &nnp));
				if (ierr != xf_OK) return NULL;
				
				/*
				// Determine Lagrange basis. (not sure if this is right?)
				ierr = xf_Error(xf_Basis2Lagrange(UBasis, &LagBasis));
				if (ierr != xf_OK) return NULL;
				*/
				// Obtain Lagrange node locations.
				ierr = xf_Error(xf_LagrangeNodes(QBasis, POrder, NULL, NULL, &xn));
				if (ierr != xf_OK) return NULL;
				
				// Compute basis functions at Lagrange nodes.
				ierr = xf_Error(xf_EvalBasis(UBasis, UOrder, xfe_False, nnp, xn,
					xfb_Phi, &PhiU));
				if (ierr != xf_OK) return NULL;
				
				// Compute basis functions at Lagrange nodes.
				ierr = xf_Error(xf_EvalBasis(QBasis, QOrder, xfe_False, nnp, xn,
					xfb_Phi, &PhiQ));
				if (ierr != xf_OK) return NULL;
				
				// Determine number of triangles in element.
				ierr = xf_Error(px_Basis2Tri(UBasis, POrder, &nt, &T0));
				if (ierr != xf_OK) return NULL;
			}
			
			// Geometry nodes
			igN = EG.Node[elem];
			// Allocate xyN...
			// Loop through nodes
			for (k=0; k<nnq; k++) {
				for (d=0; d<dim; d++) {
					// Pull of node coordinates.
					xyN[k*dim+d] = All->Mesh->Coord[igN[k]][d];
				}
			}
			
			// Calculate global coordinates of subdivision nodes.
			xf_MxM_Set(PhiQ->Phi, xyN, nnp, nnq, dim, xyG);
			// Interpolate the state onto those nodes.
			xf_MxM_Set(PhiU->Phi, EU, nnp, nnu, sr, u0);
			
			/*if (elem < 6) {
				printf("elem=%i\n", elem);
				printf("  nnp=%i, nnq=%i, nnu=%i\n", nnp, nnq, nnu);
				for (i=0; i<nnp; i++) {
					//printf("  x[%i]=%+.4e, y[%i]=%+.4e\n", i,xyG[i*dim],i,xyG[i*dim+1]);
					printf("  ");
					for (k=0; k<sr; k++) {
						printf("u[%i,%i]=%+.4e, ", i,k,u0[i*sr+k]);
					}
					printf("\n");
				}
			}*/
			
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
	dims[0] = nx;
	// Number of coordinates
	dims[1] = dim;
	// Build the coordinate array.
	PyX = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, *X);
	
	// Number of states (from EqnSet)
	dims[1] = sr;
	// Build the state array.
	PyU = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, *u);
	
	// Number of triangles
	dims[0] = nT;
	// Number of nodes per shape
	dims[1] = TRINODE;
	// Build the triangulation.
	PyT = PyArray_SimpleNewFromData(2, dims, NPY_INT, *T);
	
	// Output
	return Py_BuildValue("OOO", PyX, PyU, PyT);
}


// Function to get triangulation for an element
/*
PURPOSE:
  Get triangulation for one element and save it to a matrix if given.
  
CALL:
  ierr = px_Basis2Tri(UBasis, POrder, &nt, &T)

INPUTS:
  UBasis : xfe_BasisType
  POrder : interpolation order or number of 1D subdivisions of element

OUTPUTS:
  nt : number of triangles
  T  : (optional) [nT x 3] matrix with `nt` more rows assigned
*/
extern int
px_Basis2Tri(enum xfe_BasisType UBasis, int POrder, int *pnt, int **pT)
{
	int ierr, nt, i, j, k, ii;
	int **kk;
	int *T;
	int n = POrder;
	int nn;
	
	// Determine number of triangles in element.
	switch (UBasis) {
	case xfe_TriLagrange:
	case xfe_TriHierarch:
		// Number of triangles
		nt = (n*(n+1) + n*(n-1))/2;
		// Number of nodes
		nn = (n+1)*(n+2)/2;
		break;
	case xfe_QuadLagrange:
		// Number of triangles
		nt = 2*n*n;
		// Number of nodes
		nn = (n+1)*(n+1);
		break;
	default:
		ierr = xf_Error(xf_NOT_SUPPORTED);
		return ierr;
		break;
	}
	
	// Assign the number of triangles to output.
	(*pnt) = nt;
	
	// Check for triangles.
	if (pT == NULL) return xf_OK;
	// Allocate the triangles.
	ierr = xf_Error(xf_ReAlloc((void **) pT, 3*nt, sizeof(real)));
	if (ierr != xf_OK) return ierr;
	// Assign the pointer.
	T = (*pT);
	
	// Allocate counters.
	ierr = xf_Error(xf_Alloc2((void ***) &kk, nn, nn, sizeof(int)));
	if (ierr != xf_OK) return ierr;
	
	// Determine the network of triangles.
	switch (UBasis) {
	case xfe_TriLagrange:
	case xfe_TriHierarch:
		// Triangle count
		k = 0;
		// Loop through rows of triangles.
		for (i=0; i<n+1; i++) {
			// Loop through "columns" of triangles.
			for (j=0; j<n-i+2; j++) {
				// Vertex
				kk[i][j] = k;
				// Triangle count
				k++;
			} // j
		} // i
		
		// Triangle count
		k = 0;
		// Loop through rows of triangles.
		for (i=0; i<n; i++) {
			// Loop through "columns" of triangles.
			for (j=0; j<n-i+1; j++) {
				// Save the node indices.
				T[3*k+0] = kk[i][j];
				T[3*k+1] = kk[i+1][j];
				T[3*k+2] = kk[i][j+1];
				// Triangle count
				k++;
			} // j
		} // i
		// Loop through the remaining triangles.
		for (i=0; i<n-1; i++) {
			// Loop through the "columns".
			for (j=0; j<n-i; j++) {
				// Save the node indices.
				T[3*k+0] = kk[i][j+1];
				T[3*k+1] = kk[i+1][j];
				T[3*k+2] = kk[i+1][j+1];
				// Triangle count
				k++;
			} // j
		} // i
		
		break;
	case xfe_QuadLagrange:
		// Triangle count
		k = 0;
		// Loop through "columns" of triangles.
		for (j=0; j<n; j++) {
			// Loop through "rows" of triangles.
			for (i=0; i<n; i++) {
				// Lower left corner
				ii = j*(n+1) + i;
				// Save the vertices for the first triangle.
				T[3*k+0] = ii;
				T[3*k+1] = ii+1;
				T[3*k+2] = ii+n+1;
				// Increase triangle count.
				k++;
				// Save the vertices for the second triangle.
				T[3*k+0] = ii+1;
				T[3*k+1] = ii+n+2;
				T[3*k+2] = ii+n+1;
				// Increase triangle count,
				k++;
			} // i
		} // j
		
		break;
	default:
		ierr = xf_Error(xf_NOT_SUPPORTED);
		return ierr;
		break;
	}
	
	// Check for consistency.
	if (k != nt) return xf_INCOMPATIBLE;
	
	return xf_OK;
}

