"""Module for ingesting FAERS data files from configured source locations."""

import logging
from pyspark.sql import DataFrame
from pyspark import pipelines as dp
from pvprototypes_faers.constants import Constants
from pvprototypes_faers.data_ingestion.raw_data_ingestion import RawDataIngestion


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] => %(message)s")
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)

# --- constants ---

constants = Constants()
STAGING_CATALOG = spark.conf.get("staging_catalog")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
BRONZE_CATALOG = spark.conf.get("bronze_catalog")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
SCHEMA = spark.conf.get("schema")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

# --- helper functions ---


def _process_staging_table_stream(spark, table: str) -> DataFrame:
    logger.info(f"[_process_staging_table_stream|in] ({table})")
    df: DataFrame = spark.readStream.table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_{table}")  # pyright: ignore[reportUndefinedVariable]
    result: DataFrame = RawDataIngestion().process({"table": table, "dataframe": df})
    logger.info(f"[_process_staging_table_stream|out] => {result}")
    return result


# ----------------------------------- bronze layer

dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.demo", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.demo")
def bronze_demo() -> DataFrame:
    """Read and ingest bronze FAERS demo data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS demo data.
    """
    logger.info("[bronze_demo|in]")
    df: DataFrame = _process_staging_table_stream(spark, "demo")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_demo|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.drug", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.drug")
def bronze_drug() -> DataFrame:
    """Read and ingest bronze FAERS drug data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS drug data.
    """
    logger.info("[bronze_drug|in]")
    df: DataFrame = _process_staging_table_stream(spark, "drug")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_drug|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.outc", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.outc")
def bronze_outc() -> DataFrame:
    """Read and ingest bronze FAERS outc data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS outc data.
    """
    logger.info("[bronze_outc|in]")
    df: DataFrame = _process_staging_table_stream(spark, "outc")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_outc|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.rpsr", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.rpsr")
def bronze_rpsr() -> DataFrame:
    """Read and ingest bronze FAERS rpsr data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS rpsr data.
    """
    logger.info("[bronze_rpsr|in]")
    df: DataFrame = _process_staging_table_stream(spark, "rpsr")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_rpsr|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.ther", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.ther")
def bronze_ther() -> DataFrame:
    """Read and ingest bronze FAERS ther data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS ther data.
    """
    logger.info("[bronze_ther|in]")
    df: DataFrame = _process_staging_table_stream(spark, "ther")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_ther|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.indi", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.indi")
def bronze_indi() -> DataFrame:
    """Read and ingest bronze FAERS indi data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS indi data.
    """
    logger.info("[bronze_indi|in]")
    df: DataFrame = _process_staging_table_stream(spark, "indi")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_indi|out] => {df}")
    return df


dp.create_streaming_table(f"`{BRONZE_CATALOG}`.{SCHEMA}.reac", partition_cols=["period"])


@dp.append_flow(target=f"`{BRONZE_CATALOG}`.{SCHEMA}.reac")
def bronze_reac() -> DataFrame:
    """Read and ingest bronze FAERS reac data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing bronze FAERS reac data.
    """
    logger.info("[bronze_reac|in]")
    df: DataFrame = _process_staging_table_stream(spark, "reac")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[bronze_reac|out] => {df}")
    return df
