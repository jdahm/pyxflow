#ifndef _PX_GEOM_H
#define _PX_GEOM_H

PyObject *
px_CreateGeom(PyObject *self, PyObject *args);
char doc_CreateGeom[] =
"Create an empty *xf_Geom* instance.\n"
"\n"
":Call:\n"
"   >>> G = px.CreateGeom()\n"
"\n"
":Parameters:\n"
"   ``None``\n"
"\n"
":Returns:\n"
"   *G*: :class:`int`\n"
"       Pointer to empty *xf_Geom* instance\n";


PyObject *
px_ReadGeomFile(PyObject *self, PyObject *args);
char doc_ReadGeomFile[] =
"Create an *xf_Geom* instance from a *.geom* file.\n"
"\n"
":Call:\n"
"   >>> G = px.ReadGeomFile(fname)\n"
"\n"
":Parameters:\n"
"   *fname*: :class:`str`\n"
"       Name of geometry file to read\n"
"\n"
":Returns:\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom* instance\n";


PyObject *
px_WriteGeomFile(PyObject *self, PyObject *args);
char doc_WriteGeomFile[] =
"Write an existing *xf_Geom* to ASCII file.\n"
"\n"
":Call:\n"
"   >>> px.WriteGeomFile(G, fname)\n"
"\n"
":Parameters:\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom* instance\n"
"   *fname*: :class:`str`\n"
"       Name of mesh file to read\n"
"\n"
":Returns:\n"
"   ``None``\n"
":Examples:\n"
"   This API function can be used to write a geometry file.\n"
"\n"
"       >>> Geom = pyxflow.xf_Mesh('naca.geom')\n"
"       >>> px.WriteGeomFile(Geom._ptr, 'naca_copy.geom')\n"
"\n"
"   In addition, it can be used to write the geometry from a solution.\n"
"\n"
"       >>> All = pyxflow.xf_All('naca_adapt_0.xfa')\n"
"       >>> px.WriteGeomFile(All.Geom._ptr, 'naca_copy.geom')\n";


PyObject *
px_nGeomComp(PyObject *self, PyObject *args);
char doc_nGeomComp[] =
"Get the number of boundary geometry components from an *xf_Geom* instance.\n"
"\n"
":Call:\n"
"   >>> nComp = px.nGeomComp(G)\n"
"\n"
":Parameters:\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom* instance\n"
"\n"
":Returns:\n"
"   *nComp*: :class:`int`\n"
"       Number of geometry components\n";


PyObject *
px_GeomComp(PyObject *self, PyObject *args);
char doc_GeomComp[] =
"Get basic information from an *xf_GeomComp* geometry component.\n"
"\n"
":Call:\n"
"   >>> (Name, Type, BFGTitle, D) = px.GeomComp(G, i)\n"
"\n"
":Parameters:\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom* instance\n"
"   *i*: :class:`int`\n"
"       Index of geometry component to read\n"
"\n"
":Returns:\n"
"   *Name*: :class:`str`\n"
"       Name of geometry component\n"
"   *Type*: :class:`str`\n"
"       Geometry component type\n"
"   *BFGTitle*: :class:`str`\n"
"       Name of boundary group to which geometry is applied\n"
"   *D*: :class:`dict` or ``None``\n"
"       Spline data if appropriate\n";


PyObject *
px_DestroyGeom(PyObject *self, PyObject *args);
char doc_DestroyGeom[] =
"Destroy an *xf_Geom* instance and free memory.\n"
"\n"
":Call:\n"
"   >>> px.DestroyGeom(G)\n"
"\n"
":Parameters:\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom* instance\n"
"\n"
":Returns:\n"
"   ``None``\n";

#endif
