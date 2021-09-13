.. _api_schema:

Database Schema
===============

The synaptic physiology database is provided as sqlite files that can be queried using many relational database tools. Although it is possible to read this dataset using standard SQL queries, we also provide an sqlalchemy model that implements a richer interface to the database. The API reference below is derived from the sqlalchemy model classes, but also doubles as a description of the relational database schema.

..
  generation code:
  import aisynphys.database.schema as s
  for c in s.ORMBase.__subclasses__():
     print(f"{c.__name__}\n{'-'*len(c.__name__)}\n\n.. autoclass:: aisynphys.database.schema.{c.__name__}\n\n")

Metadata
--------

.. autoclass:: aisynphys.database.schema.Metadata


Slice
-----

.. autoclass:: aisynphys.database.schema.Slice


Experiment
----------

.. autoclass:: aisynphys.database.schema.Experiment


Electrode
---------

.. autoclass:: aisynphys.database.schema.Electrode


Cell
----

.. autoclass:: aisynphys.database.schema.Cell


Pair
----

.. autoclass:: aisynphys.database.schema.Pair


Intrinsic
---------

.. autoclass:: aisynphys.database.schema.Intrinsic


Morphology
----------

.. autoclass:: aisynphys.database.schema.Morphology


SyncRec
-------

.. autoclass:: aisynphys.database.schema.SyncRec


Recording
---------

.. autoclass:: aisynphys.database.schema.Recording


PatchClampRecording
-------------------

.. autoclass:: aisynphys.database.schema.PatchClampRecording


MultiPatchProbe
---------------

.. autoclass:: aisynphys.database.schema.MultiPatchProbe


TestPulse
---------

.. autoclass:: aisynphys.database.schema.TestPulse


StimPulse
---------

.. autoclass:: aisynphys.database.schema.StimPulse


StimSpike
---------

.. autoclass:: aisynphys.database.schema.StimSpike


PulseResponse
-------------

.. autoclass:: aisynphys.database.schema.PulseResponse


Baseline
--------

.. autoclass:: aisynphys.database.schema.Baseline


Synapse
-------

.. autoclass:: aisynphys.database.schema.Synapse


AvgResponseFit
--------------

.. autoclass:: aisynphys.database.schema.AvgResponseFit


PolySynapse
-----------

.. autoclass:: aisynphys.database.schema.PolySynapse


PulseResponseFit
----------------

.. autoclass:: aisynphys.database.schema.PulseResponseFit


PulseResponseStrength
---------------------

.. autoclass:: aisynphys.database.schema.PulseResponseStrength


Dynamics
--------

.. autoclass:: aisynphys.database.schema.Dynamics


SynapsePrediction
-----------------

.. autoclass:: aisynphys.database.schema.SynapsePrediction


RestingStateFit
---------------

.. autoclass:: aisynphys.database.schema.RestingStateFit


GapJunction
-----------

.. autoclass:: aisynphys.database.schema.GapJunction


CorticalCellLocation
--------------------

.. autoclass:: aisynphys.database.schema.CorticalCellLocation


CorticalSite
------------

.. autoclass:: aisynphys.database.schema.CorticalSite


PatchSeq
--------

.. autoclass:: aisynphys.database.schema.PatchSeq


SynapseModel
------------

.. autoclass:: aisynphys.database.schema.SynapseModel


Pipeline
--------

.. autoclass:: aisynphys.database.schema.Pipeline


