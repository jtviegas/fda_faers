"""Module for loading and managing spaCy NLP models.

This module provides a singleton class for loading and accessing spaCy language models,
with automatic installation of scispacy models if not already present.
"""

import subprocess
import sys

import spacy
from types import MappingProxyType
from pvprototypes_faers.nlp.model import Model


_SCISPACY_MODELS: MappingProxyType[str, str] = MappingProxyType(
    {
        "en_core_sci_sm": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
    }
)


class NlpModel(Model):
    """spaCy NLP model implementation.

    Provides lazy loading of spaCy models with automatic installation
    of scispacy models from remote sources if not available locally.
    """

    def __init__(self):
        """Initialize the NlpModel singleton with the en_core_sci_sm model."""
        self.__nlp = self._load_model("en_core_sci_sm")

    @staticmethod
    def _load_model(model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            url = _SCISPACY_MODELS[model_name]
            subprocess.check_call([sys.executable, "-m", "pip", "install", url])  # noqa: S603
            return spacy.load(model_name)

    @property
    def model(self):
        """Get the loaded spaCy language model.

        Returns:
            Language: The loaded spaCy language model instance.
        """
        return self.__nlp
