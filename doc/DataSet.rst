*******************************************************
Interface to XFlow *xf_DataSet*, :mod:`pyxflow.DataSet`
*******************************************************

.. automodule:: pyxflow.DataSet

Interface for *xf_DataSet* structs
==================================

.. autoclass:: pyxflow.DataSet.xf_DataSet
    
.. autoclass:: pyxflow.DataSet.xf_Data


Vectors, Vector Groups, and Arrays
==================================

.. autoclass:: pyxflow.DataSet.xf_VectorGroup
    :members: Plot, GetVector
    
.. autoclass:: pyxflow.DataSet.xf_Vector
    :members: Plot

.. autoclass:: pyxflow.DataSet.xf_GenArray


API Functions for *xf_Geom*
===========================

Several of the underlying API function apply directly to
:class:`pyxflow.DataSet.xf_DataSet` instances, XFlow *xf_DataSet* instances, and
their respective members.  The functions
above are primarily built on these functions, but an advanced user may wish to
make other uses of these low-level functions.  For the source code to these
Python/C interface functions, see `px_DataSet.c`.

.. automodule:: pyxflow._pyxflow
    :members: CreateDataSet, DestroyDataSet, ReadDataSetFile
