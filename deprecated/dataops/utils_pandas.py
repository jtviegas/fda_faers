"""Utilities for pandas DataFrame operations."""


def validate_schema(df, expected: dict) -> None:
    """Validate that DataFrame schema matches expected column types.

    Args:
        df: DataFrame to validate.
        expected: Dict mapping column names to expected dtype strings.

    Raises:
        ValueError: If schema has missing columns or type mismatches.
    """
    actual = df.dtypes.astype(str).to_dict()
    mismatches = {col: (expected[col], actual.get(col)) for col in expected if actual.get(col) != expected[col]}
    missing = [col for col in expected if col not in actual]
    if missing or mismatches:
        raise ValueError(f"Schema mismatch — missing: {missing}, wrong types: {mismatches}")
