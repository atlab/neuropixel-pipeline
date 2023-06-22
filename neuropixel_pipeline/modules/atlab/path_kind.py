from __future__ import annotations

from pydantic import BaseModel, computed_field
from enum import Enum
from pathlib import Path


class PathKind(str, Enum):
    """
    Filepath Kind
    """
    SESSION = 'session'
    CLUSTERING = 'clustering'
    CURATION = 'curation'


class PathData(BaseModel):
    """
    Filepath Abstraction
    For at-lab the data may be stored in several places, but all with a common part of the path
    """
    kind: PathKind
    path: Path

    @computed_field
    def static(self, v) -> Path:
        """
        The part of the path that doesn't change
        """
        parts = self.path.parts
        index = parts.index("raw") # raw is specific to atlab
        return Path().joinpath(*parts[index + 1 :])
    
    @staticmethod
    def specific_to_generic(specific_path: Path) -> Path:
        """
        Generates a generic path from a specific path
        """
        parts = Path(specific_path).parts
        index = parts.index('Mouse')
        return Path('/raw').joinpath(*parts[index:])
    
    @staticmethod
    def from_specific(specific_path: Path, kind: PathKind) -> PathKind:
        """
        Generates a generic path from a specific path
        """
        return PathData(path=PathData.specific_to_generic(specific_path), kind=kind)

    def specify(self, base_dir: Path) -> Path:
        """
        Specify a real location using a base directory
        """
        return Path(base_dir) / self.static
    
    def normalized(self, base_dir: Path) -> Path:
        """
        Handles path kind specific nuances

        This is the function that should generally be used
        """
        specific_path = self.specify(base_dir=base_dir)
        if self.kind is PathKind.SESSION:
            return specific_path.parent
        elif self.kind is PathKind.CLUSTERING:
            return specific_path
        elif self.kind is PathKind.CURATION:
            return specific_path
        else:
            raise NotImplementedError(f"this is not implemented for this PathKind: {self.kind}")
