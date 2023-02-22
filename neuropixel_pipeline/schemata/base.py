"""
base, upstream schema for the neuropixel schemas

Potentially a temporary schema as a placeholder for other upstream schemas,
however this might instead transform into a wrapping schema that centralizes all upstream outside connections
"""

import datajoint as dj

schema = dj.schema('neuropixel_base')

@schema
class Session(dj.Manual):
    definition = """
    # Session: table connection
    session_id : int unsigned # Session primary key
    """

@schema
class SkullReference(dj.Lookup):
    definition = """
    # SkullReference, not actually sure where this table comes from in the element
    reference: int
    """