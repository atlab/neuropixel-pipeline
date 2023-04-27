"""
Currently defines the manual ingest points that also get used for the datajoint pipeline.
"""

from pydantic import BaseModel

# Currently, a lot of the work for what element_array_ephys does in the classmethods and make
# functions can be pulled out and unified under more modular and DRYer reader types.
# This is much like what it already does, but properly kept separate from the rest of the pipeline
# code.

class ProbeMeta(BaseModel):
    ...

class ConfigLabview(ProbeMeta):
    ...