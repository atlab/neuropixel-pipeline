from pydantic import validate_call
from pathlib import Path

from .probe_setup import probe_setup
from .session_search import ScanKey, get_session_path

@validate_call
def main(scan_key: ScanKey, base_dir: Path = None, setup: bool = None):
    if setup:
        probe_setup()

    session_path = get_session_path(scan_key, base_dir=base_dir)

    raise NotImplementedError("Not implemented to this point yet")

if __name__ == '__main__':
    main()