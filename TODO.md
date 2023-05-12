- Full populate
    - populate for all the metadata + pykilosort
    - special consideration for reading the metadata from the h5 files may be needed (at least as a temporary measure until that metadata is extracted into a separate file alongside the full bin file)
    - most code should be separate from the populate but the make functions do need to be, well, functional

- Define ingest
    - Parts at which manual ingest is needed, and at which I can ingest directly from the files (after the paths are inserted first of course)
    - These can easily be defined with pydantic Models!