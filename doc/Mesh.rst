*************************************************
Interface to XFlow *xf_Mesh*, :mod:`pyxflow.Mesh`
*************************************************

.. automodule:: pyxflow.Mesh

Interface for *xf_Mesh* structs
===============================

.. autoclass:: pyxflow.Mesh.xf_Mesh
    :members: Plot
    
.. autoclass:: pyxflow.Mesh.xf_BFaceGroup

.. autoclass:: pyxflow.Mesh.xf_ElemGroup

API Functions for *xf_Mesh*
===========================

Several of the underlying API function apply directly to
:class:`pyxflow.Mesh.xf_Mesh` and XFlow *xf_Mesh* instances.  The functions above
are primarily built on these functions, but an advanced user may wish to make
other uses of these low-level functions.  For the source code to these Python/C
interface functions, see `px_Mesh.c`.

.. automodule:: pyxflow._pyxflow
    :members: CreateMesh, DestroyMesh, ReadGriFile, WriteGriFile, GetNodes,
        nBFaceGroup, BFaceGroup, nElemGroup, ElemGroup
