#ifndef _PX_ALL_H
#define _PX_ALL_H

/***************************************************/
PyObject *
px_CreateAll(PyObject *self, PyObject *args);

char doc_CreateAll[] =
"Create empty *xf_All* struct.\n"
"\n"
":Call:\n"
"   >>> A = px.CreateAll(DefaultFlag)\n"
"\n"
":Parameters:\n"
"   *DefaultFlag*: :class:`bool`\n"
"       Whether or not to set default parameters\n"
"\n"
":Returns:\n"
"   *A*: :class:`int`"
"       Pointer to empty *xf_All*\n";

/***************************************************/
PyObject *
px_DestroyAll(PyObject *self, PyObject *args);

char doc_DestroyAll[] =
"Destroy *xf_All* struct and its contents.\n"
"\n"
":Call:\n"
"   >>> px.DestroyAll(A)\n"
"\n"
":Parameters:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All* struct\n";

/***************************************************/
PyObject *
px_ReadAllInputFile(PyObject *self, PyObject *args);

char doc_ReadAllInputFile[] =
"Create an *xf_All* struct by reading an input file.\n"
"\n"
":Call:\n"
"   >>> A = px.ReadAllInputFile(InputFile, DefaultFlag)\n"
"\n"
":Parameters:\n"
"   *InputFile*: :class:`str`\n"
"       Name of file to read\n"
"   *DefaultFlag*: :class:`bool`\n"
"       Whether or not to set default parameters\n"
"\n"
":Returns:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All* struct\n";

/***************************************************/
PyObject *
px_ReadAllBinary(PyObject *self, PyObject *args);

char doc_ReadAllBinary[] =
"Read an XFlow ``'.xfa'`` file and return the pointer.\n"
"\n"
":Call:\n"
"   >>> A = px.ReadAllBinary(XfaFile, DefaultFlag)\n"
"\n"
":Parameters:\n"
"   *XfaFile*: :class:`str`\n"
"       Name of file to read from\n"
"   *DefaultFlag*: :class:`bool`\n"
"       Whether or not to use defaults internally\n"
"\n"
":Returns:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All* struct\n";

/***************************************************/
PyObject *
px_WriteAllBinary(PyObject *self, PyObject *args);

char doc_WriteAllBinary[] =
"Write the contents of an XFlow *xf_All* to file.\n"
"\n"
":Call:\n"
"   >>> px.WriteAllBinary(A, fname)\n"
"\n"
":Parameters:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All* struct\n"
"   *fname*: :class:`str`\n"
"       Name of ``.xfa`` file to create\n"
"\n"
":Returns:\n"
"   ``None``\n"
"\n"
":Examples:\n"
"   This API function uses the hidden field *All._ptr*.\n"
"       >>> All = pyxflow.xf_All(fname='naca_Adapt_0.xfa')\n"
"       >>> px.WriteAllBinary(All._ptr, 'test.xfa')\n";

/***************************************************/
PyObject *
px_GetAllMembers(PyObject *self, PyObject *args);

char doc_GetAllMembers[] =
"Get pointers to members of *xf_All*.\n"
"\n"
":Call:\n"
"   >>> M, G, DS, P, ES = px.GetAllMembers(A)\n"
"\n"
":Parameters:\n"
"   *A*: :class:`int`\n"
"       Pointer to *xf_All*\n"
"\n"
":Returns:\n"
"   *M*: :class:`int`\n"
"       Pointer to *xf_Mesh*\n"
"   *G*: :class:`int`\n"
"       Pointer to *xf_Geom*\n"
"   *DS*: :class:`int`\n"
"       Pointer to *xf_DataSet*\n"
"   *P*: :class:`int`\n"
"       Pointer to *xf_Param*\n"
"   *ES*: :class:`int`\n"
"       Pointer to *xf_EqnSet*\n";


#endif
