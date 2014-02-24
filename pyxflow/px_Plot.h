#ifndef _PX_PLOT_H
#define _PX_PLOT_H


PyObject *
px_MeshPlotData(PyObject *self, PyObject *args);
char doc_MeshPlotData[] =
"Calculate mesh data for plotting\n"
"\n"
":Call:\n"
"   >>> x, y, C = px.MeshPlotData(M, xmin, xmax, order)\n"
"\n"
":Parameters:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* structure\n"
"   *xmin*: :class:`list`\n"
"       List of minimum coordinates for each dimension\n"
"   *xmax*: :class:`list`\n"
"       List of maximum coordinates for each dimension\n"
"   *order*: :class:`int`\n"
"       Plot order\n"
"\n"
":Returns:\n"
"   *x*: :class:`numpy.array`\n"
"       Vector of nodal *x*-coordinates\n"
"   *y*: :class:`numpy.array`\n"
"       Vector of nodal *y*-coordinates\n"
"   *C*: :class:`numpy.array`\n"
"       Connectivity data; segment ``f`` has endpoints\n"
"       ``(x[C[f]],y[C[f]])``  and ``(x[C[f+1]],y[C[f+1]])``.\n";

PyObject *
px_ScalarPlotData(PyObject *self, PyObject *args);
char doc_ScalarPlotData[] =
"Calculate scalar data for plotting\n"
"\n"
":Call:\n"
"   >>> x, y, T, u = px.MeshPlotData(U, M, E, Name, xmin, xmax, order)\n"
"\n"
":Parameters:\n"
"   *U*: :class:`int`\n"
"       Pointer to *xf_Vector* structure\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh* structure\n"
"   *E*: :class:`int`\n"
"       Pointer to *xf_EqnSet* structure\n"
"   *Name*: :class:`str`\n"
"       Name of scalar to plot. If ``None``, first vector entry is used.\n"
"   *xmin*: :class:`list`\n"
"       List of minimum coordinates for each dimension\n"
"   *xmax*: :class:`list`\n"
"       List of maximum coordinates for each dimension\n"
"   *order*: :class:`int`\n"
"       Plot order. If ``None``, vector solution order is used.\n"
"\n"
":Returns:\n"
"   *x*: :class:`numpy.array` (*np*)\n"
"       Vector of nodal *x*-coordinates\n"
"   *y*: :class:`numpy.array` (*np*)\n"
"       Vector of nodal *y*-coordinates\n"
"   *T*: :class:`numpy.array` (*np*, *3*)\n"
"       Triangulation matrix; node indices for each triangle\n"
"   *u*: :class:`numpy.array` (*np*)\n"
"       Scalar value at each node\n";


#endif
