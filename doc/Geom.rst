*************************************************
Interface to XFlow *xf_Geom*, :mod:`pyxflow.Geom`
*************************************************

.. automodule:: pyxflow.Geom

Interface for *xf_Geom* structs
===============================

.. autoclass:: pyxflow.Geom.xf_Geom
    :members: Write
    
.. autoclass:: pyxflow.Geom.xf_GeomComp

.. autoclass:: pyxflow.Geom.xf_GeomCompSpline

API Functions for *xf_Geom*
===========================

Several of the underlying API function apply directly to
:class:`pyxflow.Geom.xf_Geom` and XFlow *xf_Geom* instances.  The functions
above are primarily built on these functions, but an advanced user may wish to
make other uses of these low-level functions.  For the source code to these
Python/C interface functions, see `px_Geom.c`.

.. automodule:: pyxflow._pyxflow
    :members: CreateGeom, DestroyGeom, ReadGeomFile, WriteGeomFile, nGeomComp,
        GeomComp
