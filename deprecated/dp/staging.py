"""Module for ingesting FAERS data files from configured source locations."""

import logging
from pyspark.sql import DataFrame
from pyspark import pipelines as dp
from pvprototypes_faers.constants import Constants
from pyspark.sql.types import StructType, StructField, LongType

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
SCHEMA = spark.conf.get("schema")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
FILES_URL = spark.conf.get("faers_files_url")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

# --- helper functions ---


def _read_stream(spark, table_name: str, files_url: str) -> DataFrame:
    logger.info(f"[_read_stream|in] (table_name={table_name}, files_url={files_url})")
    result: DataFrame = (
        spark.readStream.format("cloudFiles")  # pyright: ignore[reportUndefinedVariable]
        .option("cloudFiles.format", "csv")
        .option("pathGlobFilter", f"{table_name}*.txt")
        .option("delimiter", constants.CSV_DELIMITER)
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("header", "true")
        .load(files_url)
    )
    logger.info(f"[_read_stream|out] => {result}")
    return result


# ----------------------------------- staging layer

dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_deleted")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_deleted")
def staging_deleted_flow() -> DataFrame:
    """Read and ingest deleted FAERS cases from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing deleted case IDs.
    """
    logger.info("[staging_deleted_flow|in]")
    result = (
        spark.readStream.format("cloudFiles")  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
        .option("cloudFiles.format", "csv")
        .option("pathGlobFilter", "*delete*.txt")
        .option("header", "false")
        .schema(StructType([StructField("caseid", LongType(), nullable=True)]))
        .load(FILES_URL)
    )
    logger.info(f"[staging_deleted_flow|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_demo")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_demo")
def staging_demo() -> DataFrame:
    """Read and ingest raw FAERS demo data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS demo data.
    """
    logger.info("[staging_demo|in]")
    table = "demo"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_demo|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_drug")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_drug")
def staging_drug() -> DataFrame:
    """Read and ingest raw FAERS drug data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS drug data.
    """
    logger.info("[staging_drug|in]")
    table = "drug"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_drug|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_reac")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_reac")
def staging_reac() -> DataFrame:
    """Read and ingest raw FAERS reaction data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS reaction data.
    """
    logger.info("[staging_reac|in]")
    table = "reac"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_reac|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_outc")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_outc")
def staging_outc() -> DataFrame:
    """Read and ingest raw FAERS outcome data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS outcome data.
    """
    logger.info("[staging_outc|in]")
    table = "outc"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_outc|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_indi")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_indi")
def staging_indi() -> DataFrame:
    """Read and ingest raw FAERS individual data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS indi data.
    """
    logger.info("[staging_indi|in]")
    table = "indi"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_indi|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_rpsr")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_rpsr")
def staging_rpsr() -> DataFrame:
    """Read and ingest raw FAERS rpsr data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS rpsr data.
    """
    logger.info("[staging_rpsr|in]")
    table = "rpsr"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_rpsr|out] => {result}")
    return result


dp.create_streaming_table(f"`{STAGING_CATALOG}`.{SCHEMA}.staging_ther")


@dp.append_flow(target=f"`{STAGING_CATALOG}`.{SCHEMA}.staging_ther")
def staging_ther() -> DataFrame:
    """Read and ingest raw FAERS ther data from the configured source location.

    Returns:
        DataFrame: A streaming DataFrame containing raw FAERS ther data.
    """
    logger.info("[staging_ther|in]")
    table = "ther"
    result = _read_stream(spark, table_name=table, files_url=FILES_URL)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    logger.info(f"[staging_ther|out] => {result}")
    return result
