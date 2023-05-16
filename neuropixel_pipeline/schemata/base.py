"""
base, upstream schema for the neuropixel schemas

Potentially a temporary schema as a placeholder for other upstream schemas,
however this might instead transform into a wrapping schema that centralizes all upstream outside connections
"""

import datajoint as dj

schema = dj.schema("neuropixel_base")


# TODO: connect session with upstream
@schema
class Session(dj.Manual):
    """Session key"""

    definition = """
    # Session: table connection
    session_id : int unsigned # Session primary key
    """


@schema
class SkullReference(dj.Lookup):
    """Reference area from the skull"""

    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(["Bregma", "Lambda"])
