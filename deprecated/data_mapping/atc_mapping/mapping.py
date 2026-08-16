"""Module for ATC class data structures and the AtcMapping abstract interface.

This module provides classes for representing ATC (Anatomical Therapeutic Chemical)
classifications and the abstract DataFrame-level mapping interface (AtcMapping).
"""

from abc import ABC, abstractmethod
from attr import dataclass
from pyspark.sql import DataFrame
from typing import ClassVar
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


@dataclass(frozen=True)
class AtcClass:
    """
    Data class depicting the concept of an ATC class.
    """

    id: str
    name: str
    class_type: str

    def to_dict(self) -> dict:
        """
        Convert the AtcClass instance to a dictionary.

        Returns:
            A dictionary with keys 'id', 'name', and 'class_type'.
        """
        return {"id": self.id, "name": self.name, "class_type": self.class_type}

    @staticmethod
    def from_dict(d) -> "AtcClass":
        """
        Create an AtcClass instance from a dictionary.

        Args:
            d: A dictionary with keys 'id', 'name', and 'class_type'.

        Returns:
            An AtcClass instance.
        """
        return AtcClass(id=d["id"], name=d["name"], class_type=d["class_type"])


class AtcMapping(ABC):
    """Abstract base class for DataFrame-level ATC code mapping.

    Unlike ``Mapping``, which operates term-by-term, ``AtcMapping`` receives
    a full PySpark DataFrame and returns it decorated with an additional column
    containing the resolved ATC codes. Concrete implementations decide how the
    underlying LLM or API is called (e.g. Databricks ``ai_query``, REST API).
    """

    ATC_CODE_SCHEMA: ClassVar[ArrayType] = ArrayType(
        StructType(
            [
                StructField("id", StringType(), False),
                StructField("name", StringType(), False),
                StructField("class_type", StringType(), False),
            ]
        )
    )

    def __init__(self, term_col: str = "term", output_col: str = "atc_codes") -> None:
        """Initialise with configurable column names.

        Args:
            term_col: Name of the DataFrame column that contains drug/ingredient terms.
            output_col: Name of the column to be added with the ATC codes result.
        """
        self._term_col = term_col
        self._output_col = output_col

    @abstractmethod
    def decorate(self, df: DataFrame) -> DataFrame:
        """Decorate a DataFrame with ATC codes.

        Args:
            df: Input DataFrame containing a column with drug/ingredient terms
                (column name is controlled by ``term_col`` constructor argument).

        Returns:
            The input DataFrame with an additional column (``output_col``) holding
            the ATC codes resolved for each term.
        """
