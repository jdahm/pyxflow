#ifndef _PX_MESH_H
#define _PX_MESH_H


/***************************************************/
PyObject *
px_CreateMesh(PyObject *self, PyObject *args);
char doc_CreateMesh[] =
"Create an empty *xf_Mesh* instance.\n"
"\n"
":Call:\n"
"   >>> M = px.CreateMesh()\n"
"\n"
":Parameters:\n"
"   ``None``\n"
"\n"
":Returns:\n"
"   *M*: :class:`int`\n"
"       Pointer to empty *xf_Mesh* instance\n";


PyObject *
px_ReadGriFile(PyObject *self, PyObject *args);
char doc_ReadGriFile[] =
"Create a mesh from a *.gri* file.\n"
"\n"
":Call:\n"
"   >>> M = px.ReadGriFile(fname)\n"
"\n"
":Parameters:\n"
"   *fname*: :class:`str`\n"
"       Name of mesh file to read\n"
"\n"
":Returns:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n";


PyObject *
px_WriteGriFile(PyObject *self, PyObject *args);
char doc_WriteGriFile[] =
"Write an existing mesh to ASCII file.\n"
"\n"
":Call:\n"
"   >>> px.WriteGriFile(M, fname)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"   *fname*: :class:`str`\n"
"       Name of mesh file to read\n"
"\n"
":Returns:\n"
"   ``None``\n"
":Examples:\n"
"   This API function can be used to write a mesh file from a mesh.\n"
"\n"
"       >>> Mesh = pyxflow.xf_Mesh('naca_quad.gri')\n"
"       >>> px.WriteGriFile(Mesh._ptr, 'naca_copy.gri')\n"
"\n"
"   In addition, it can be used to write the mesh of a solution to file.\n"
"\n"
"       >>> All = pyxflow.xf_All('naca_adapt_0.xfa')\n"
"       >>> px.WriteGriFile(All.Mesh._ptr, 'naca_copy.gri')\n";


PyObject *
px_GetNodes(PyObject *self, PyObject *args);
char doc_GetNodes[] =
"Read the nodes and their coordinates from an *xf_Mesh* instance.\n"
"\n"
":Call:\n"
"   >>> (Dim, nNode, Coord) = px.GetNodes(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   *Dim*: :class:`int`; ``1``, ``2``, or ``3``\n"
"       Dimension of the mesh\n"
"   *nNode*: :class:`int`\n"
"       Number of nodes in mesh\n"
"   *Coord*: :class:`numpy.array`; (*nNode*, *Dim*)\n"
"       Matrix of node coordinates\n";


PyObject *
px_nBFaceGroup(PyObject *self, PyObject *args);
char doc_nBFaceGroup[] =
"Get the number of boundary face groups from an *xf_Mesh* instance.\n"
"\n"
"This function, in `px_Mesh.c`, is an excellent example of a very simple\n"
"pyXFlow API function.\n"
"\n"
":Call:\n"
"   >>> nBG = px.nBFaceGroup(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   *nBG*: :class:`int`\n"
"       Number of boundary face groups in mesh\n";


PyObject *
px_BFaceGroup(PyObject *self, PyObject *args);
char doc_BFaceGroup[] =
"Get basic information from a boundary face group.\n"
"\n"
":Call:\n"
"   >>> (Title, nBFace, BG) = px.BFaceGroup(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   *Title*: :class:`str`\n"
"       Name of the boundary group\n"
"   *nBFace*: :class:`int`\n"
"       Number of boundary faces in the group\n"
"   *BG*: :class:`int`\n"
"       Pointer to *xf_BFaceGroup*\n";


PyObject *
px_nElemGroup(PyObject *self, PyObject *args);
char doc_nElemGroup[] =
"Get the number of element groups from an *xf_Mesh* instance.\n"
"\n"
":Call:\n"
"   >>> nEG = px.nElemGroup(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   *nEG*: :class:`int`\n"
"       Number of element groups in mesh\n";


PyObject *
px_ElemGroup(PyObject *self, PyObject *args);
char doc_ElemGroup[] =
"Get basic information from a boundary face group.\n"
"\n"
":Call:\n"
"   >>> (nElem, nNode, QOrder, QBasis, Node) = px.BFaceGroup(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   *nElem*: :class:`int`\n"
"       Number of elements in the element group\n"
"   *nNode*: :class:`int`\n"
"       Number of nodes used in the element group\n"
"   *QOrder*: :class:`int`\n"
"       Interpolation order for mesh elements\n"
"   *QBasis*: :class:`str`\n"
"      Name of basis used for elements of this group\n"
"   *Node*: :class:`numpy.array`, (*nNode*, *n*)\n"
"       Indices of nodes in each element; *n* is number of nodes per element\n";


PyObject *
px_DestroyMesh(PyObject *self, PyObject *args);
char doc_DestroyMesh[] =
"Destroy an *xf_Mesh* instance and free memory.\n"
"\n"
":Call:\n"
"   >>> px.DestroyMesh(M)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* instance\n"
"\n"
":Returns:\n"
"   ``None``\n";

#endif
