"""Pydantic adapters for working with file-loaded datajoint store data.

Connected to the stores configs in schemata (only ephys currently).

Intended for use with the filepath datajoint protocol (as opposed to blob protocol).
"""

# TODO: Need a utility class for file loading, because this will be common.

# TODO: Adapters will read the data from a filepath (using utility class) and
#       convert to a similar but slightly more useful/descriptive pydantic model.