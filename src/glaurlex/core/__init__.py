"""Computational core of GLAURLex.

Every function operates on in-memory objects (pandas DataFrames, NetworkX graphs)
and never writes to disk. The data contract shared by all modules is:

* topic frame: one row per response, columns ``informant_id``, ``pos`` (0-indexed
  position within the informant's list) and ``type``;
* informants frame: one row per informant, keyed by ``informant_id``, plus any
  sociolinguistic variables.
"""
