***************************
Interface to XFlow *xf_All*
***************************

The :mod:`pyxflow.All` submodule contains the top-level interface for XFlow
solutions.  This primarily involves loading an instance of
:mod:`pyxflow.All.xf_All` by reading an `*.xfa` file.

Once an *xf_All* variable has been created, it is possible to reference more
specific data, such as the mesh description and scalar solution values, using
members of :class:`pyxflow.All.xf_All` that are instances of other classes.

:mod:`pyxflow.All`
===================

.. autoclass:: pyxflow.All.xf_All
    :members: Plot, Write, GetPrimalState
    
.. autoclass:: pyxflow.All.xf_EqnSet


API Functions for *xf_All*
==========================

Several of the underlying API function apply directly to
:class:`pyxflow.All.xf_All` instances.  The functions above are primarily built
on these functions, but an advanced user may wish to make other uses of these
low-level functions.  For the source code to these Python/C interface functions,
see `px_All.c`.

.. automodule:: pyxflow._pyxflow
    :members: ReadAllBinary, ReadAllInputFile, WriteAllBinary, CreateAll,
        DestroyAll, GetAllMembers
