
"""Define the standardized FAERS table bundle and mapping conversion helpers.

This module provides :class:`FaersTables`, which groups the seven FAERS
datasets used to assemble case representations.
"""

from __future__ import annotations
from dataclasses import dataclass
from datasets import Dataset


@dataclass(frozen=True)
class FaersTables:
    """The seven standardized FAERS tables used to assemble case representations."""

    demo: Dataset
    drug: Dataset
    reac: Dataset
    outc: Dataset
    rpsr: Dataset
    ther: Dataset
    indi: Dataset

    def to_dict(self) -> dict[str, Dataset]:
        """Return the seven datasets keyed by their FAERS table names."""
        return {
            "demo": self.demo,
            "drug": self.drug,
            "reac": self.reac,
            "outc": self.outc,
            "rpsr": self.rpsr,
            "ther": self.ther,
            "indi": self.indi,
        }

    @classmethod
    def from_dict(cls, tables: dict[str, Dataset]) -> FaersTables:
        """Create a table bundle from a mapping, requiring all seven table names.

        Args:
            tables: Mapping from lowercase FAERS table names to datasets.

        Returns:
            A bundle containing each required dataset; extra mapping keys are ignored.

        Raises:
            ProcessorException: If any required table name is absent.
        """
        names = ("demo", "drug", "reac", "outc", "rpsr", "ther", "indi")
        missing = [name for name in names if name not in tables]
        if missing:
            raise ValueError(f"Missing FAERS tables: {', '.join(missing)}")
        return cls(**{name: tables[name] for name in names})