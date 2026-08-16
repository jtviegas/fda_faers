"""Entity parsing processor for extracting named entities from text."""

from typing import Any
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from tgedr_dataops_abs.processor import Processor
from pvprototypes_faers.nlp.model_singleton import ModelSingleton


class EntityParsing(Processor):
    """Extract named entities from text using an NLP model."""

    CONTEXT_KEY_TEXT = "text"

    def process(self, context: dict[str, Any] | None = None) -> list[str]:
        """Extract named entities from text using NLP model.

        Parameters
        ----------
        context : dict[str, Any] | None
            Context dictionary containing the text to process under the
            CONTEXT_KEY_TEXT key.

        Returns
        -------
        list[str]
            List of extracted entity texts.

        Raises
        ------
        Exception
            If context is None or does not contain the required text key.
        """
        if not context or self.CONTEXT_KEY_TEXT not in context:
            raise Exception(f"{self.CONTEXT_KEY_TEXT} must be provided in context")
        text: str = context[self.CONTEXT_KEY_TEXT]

        model = ModelSingleton().instance.model  # pyright: ignore[reportAttributeAccessIssue]
        doc = model(text)

        result: list[str] = [entity.text for entity in doc.ents]

        return result


def parse_entities(text) -> list[str] | None:
    """Extract unique named entities from text.

    Parameters
    ----------
    text : str
        The text to extract entities from.

    Returns
    -------
    list[str] | None
        List of unique extracted entity texts or None if no text is provided.
    """
    result = set()
    if not text or text.strip() == "":
        return None
    processor = EntityParsing()
    for entity in processor.process({EntityParsing.CONTEXT_KEY_TEXT: text}):
        result.add(entity)

    if len(result) == 0:
        # If no entities were found, return the original text as a single entity
        return [text.strip()]
    return list(result)


parse_entities_udf = udf(parse_entities, ArrayType(StringType()))


def parse_entities_to_comma_concatenated(text) -> str | None:
    """Extract named entities from text and return as comma-separated string.

    Parameters
    ----------
    text : str
        The text to extract entities from.

    Returns
    -------
    str | None
        Comma-separated string of unique extracted entity texts with
        normalized whitespace.
    """
    entities = parse_entities(text)
    if entities is None:
        return None
    entities = [" ".join(z.split()) for z in [x.replace(",", " ") for x in entities]]

    return ",".join(entities)


parse_entities_to_comma_concatenated_udf = udf(parse_entities_to_comma_concatenated, StringType())
