"""Module for ingesting FAERS data files from configured source locations."""

from datetime import datetime, timezone
import logging
from numpy import ceil
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark import pipelines as dp
from delta.tables import DeltaTable
from pvprototypes_faers.constants import Constants
from pvprototypes_faers.data_ingestion.drug_ingredient_entity import DrugIngredientEntity
from pvprototypes_faers.data_mapping.atc_mapping.ai_atc_mapping import AiAtcMapping
from pvprototypes_faers.data_mapping.atc_mapping.nlm_mapping import NLMMapping
from pvprototypes_faers.data_mapping.meddra_mapping import MedDRAMapping
from test.conftest import spark

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] => %(message)s")
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)

constants = Constants()

# --- constants ---

BRONZE_CATALOG = spark.conf.get("bronze_catalog")  # pyright: ignore[reportUndefinedVariable]
SILVER_CATALOG = spark.conf.get("silver_catalog")  # pyright: ignore[reportUndefinedVariable]
MEDDRA_TABLES_SCHEMA = spark.conf.get("meddra_tables_schema")  # pyright: ignore[reportUndefinedVariable]
SCHEMA = spark.conf.get("schema")  # pyright: ignore[reportUndefinedVariable]
ENTITY_MAPPING_TABLE = f"`{SILVER_CATALOG}`.{SCHEMA}.entity_atc_code"  # pyright: ignore[reportUndefinedVariable]

# --- helper functions ---


def _get_deleted_cases(spark: SparkSession) -> DataFrame:
    """Return distinct non-null case IDs from the raw deleted-cases table."""
    logger.info("[_get_deleted_cases|in]")
    result = (
        spark.read.table("raw_deleted")  # pyright: ignore[reportUndefinedVariable]
        .select("caseid")
        .where(F.col("caseid").isNotNull())
        .distinct()
    )
    logger.info("[_get_deleted_cases|out]")
    return result


def _remove_deleted_cases(spark: SparkSession, table: str) -> DataFrame:
    """Remove rows from the input table whose ``caseid`` exists in ``raw_deleted``."""
    logger.info(f"[_remove_deleted_cases|in] ({table})")

    df_deleted_cases = _get_deleted_cases(spark)
    df = spark.read.table(table)  # pyright: ignore[reportUndefinedVariable]
    initial_count = df.count()
    result = df.join(df_deleted_cases, on="caseid", how="left_anti")
    final_count = result.count()
    logger.info(f"[_remove_deleted_cases|out] ({table}) - removed {initial_count - final_count} deleted cases")
    return result


def _merge_mapped_entities(
    spark: SparkSession, df_mapped: DataFrame, mapping_table: str = ENTITY_MAPPING_TABLE
) -> None:
    """Helper function to append newly coded terms to the coded table."""
    logger.info(f"[_merge_mapped_entities|in] ({df_mapped}, {mapping_table})")
    try:
        n: int = df_mapped.cache().count()
        logger.info(f"[_merge_mapped_entities] number of coded terms to add: {n}")
        (
            DeltaTable.forName(spark, mapping_table)  # pyright: ignore[reportUndefinedVariable]
            .alias("target")
            .merge(
                source=df_mapped.alias("source"),
                condition="target.entity = source.entity",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    except Exception as ae:
        if str(ae).find("[DELTA_MISSING_DELTA_TABLE]") != -1:
            logger.info(f"table {mapping_table} not found. Creating new table with coded terms.")
            df_mapped.write.format("delta").saveAsTable(mapping_table)  # pyright: ignore[reportUndefinedVariable]
        else:
            raise ae  # noqa: TRY201
    logger.info("[_merge_mapped_entities|out]")


def _apply_mapping_strategies(df_entities: DataFrame) -> DataFrame:
    logger.info(f"[_apply_mapping_strategies|in] ({df_entities})")

    logger.info(f"[_apply_mapping_strategies] trying to map {df_entities.count()} entities with NLMMapping")
    df_mapping_1 = NLMMapping(term_col="entity", output_col="atc_code").decorate(df=df_entities)
    df_unmapped_1 = df_mapping_1.filter(F.col("atc_code").isNull()).select("entity").distinct()
    df_mapped_1 = df_mapping_1.filter(F.col("atc_code").isNotNull())
    logger.info(f"[_apply_mapping_strategies] NLMMapping mapped {df_mapped_1.count()} entities")

    logger.info(f"[_apply_mapping_strategies] trying to map {df_unmapped_1.count()} entities with AiAtcMapping")
    df_mapping_2 = AiAtcMapping(term_col="entity", output_col="atc_code").decorate(df=df_unmapped_1)
    df_mapped_2 = df_mapping_2.filter(F.col("atc_code").isNotNull())
    logger.info(f"[_apply_mapping_strategies] AiAtcMapping mapped {df_mapped_2.count()} entities")

    result = df_mapped_1.unionByName(df_mapped_2.select(df_mapped_1.columns))  # pyright: ignore[reportUndefinedVariable]
    logger.info(f"[_apply_mapping_strategies|out] => returning {result.count()} mapped entities")
    return result


def _map_entities(
    spark,
    df_entity: DataFrame,
    map_table: str = ENTITY_MAPPING_TABLE,
    n_partitions: int = 8,
    fraction: float = 0.1,
    batch_size: int = 100,
) -> None:
    logger.info(f"[_map_entities|in] ({df_entity}, {map_table}, {n_partitions}, {fraction}, {batch_size})")

    df_mapped: DataFrame = spark.createDataFrame([], schema="entity STRING")
    try:
        df_mapped: DataFrame = spark.read.table(map_table).select("entity").distinct()  # pyright: ignore[reportUndefinedVariable, reportUndefinedVariable]
    except Exception as ae:
        if str(ae).find("[TABLE_OR_VIEW_NOT_FOUND]") != -1:
            logger.warning(f"[_map_entities]Mapping table {map_table} not found. Returning empty DataFrame.")
        else:
            raise ae  # noqa: TRY201

    df_to_map: DataFrame = (
        df_entity.join(df_mapped, on="entity", how="left_anti").distinct().sample(fraction=fraction)  # pyright: ignore[reportArgumentType]
    )
    logger.info(f"size of subset to map: {df_to_map.count()}")
    ts = int(datetime.now(tz=timezone.utc).timestamp())

    n = df_to_map.count()
    if 0 < n:
        for i in range(int(ceil(n / batch_size))):
            logger.info(f"[_map_entities] processing batch {i} of unmapped entities...")
            df_batch: DataFrame = df_to_map.offset(i * batch_size).limit(batch_size).repartition(n_partitions).cache()
            try:
                df_newly_mapped: DataFrame = _apply_mapping_strategies(df_batch).withColumn(
                    "processing_time", F.lit(ts)
                )  # pyright: ignore[reportUndefinedVariable]
                _merge_mapped_entities(spark, df_newly_mapped)
            except Exception as e:
                logger.warning(f"[_map_entities] exception processing unmapped entities, resuming: {e}")

    logger.info("[_map_entities|out]")


# ----------------------------------- silver layer

dp.create_streaming_table(f"`{SILVER_CATALOG}`.{SCHEMA}.drug_ingredient_entity")


@dp.append_flow(target=f"`{SILVER_CATALOG}`.{SCHEMA}.drug_ingredient_entity")
def drug_ingredient_entity() -> DataFrame:
    """Process drug ingredient entity data.

    Returns:
        DataFrame: A streaming DataFrame containing entities from drug ingredient data.
    """
    logger.info("[drug_ingredient_entity|in]")
    result: DataFrame = DrugIngredientEntity().process(
        {
            "dataframe": spark.readStream.table(f"`{BRONZE_CATALOG}`.{SCHEMA}.drug")
            .select("drugname", "prod_ai")
            .distinct()
        }  # noqa: F821, RUF100 # pyright: ignore[reportUndefinedVariable]
    )
    # returns [ drugname, prod_ai, term, entity ]
    logger.info(f"[drug_ingredient_entity|out] => {result}")
    return result


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.drug", partition_cols=["period"])
def drug() -> DataFrame:
    """Generate ATC coding rows for uncoded ingredient terms.

    The function repeatedly fetches uncoded terms, computes ATC codes from parsed entities,
    and accumulates the generated rows with a single UTC processing timestamp.
    """
    logger.info("[drug|in]")

    df_entity: DataFrame = spark.read.table(f"`{SILVER_CATALOG}`.{SCHEMA}.drug_ingredient_entity")  # pyright: ignore[reportUndefinedVariable]
    _map_entities(
        spark=spark,
        df_entity=df_entity.select("entity").distinct(),
        map_table=ENTITY_MAPPING_TABLE,
        n_partitions=8,
        fraction=0.1,
        batch_size=100,
    )  # pyright: ignore[reportUndefinedVariable]

    df_mapped_entity: DataFrame = spark.createDataFrame(
        [],
        schema="entity STRING, atc_code ARRAY<STRUCT<id: STRING, name: STRING, class_type: STRING>>, strategy STRING",
    )  # pyright: ignore[reportUndefinedVariable]
    try:
        df_mapped_entity: DataFrame = spark.read.table(ENTITY_MAPPING_TABLE)  # pyright: ignore[reportUndefinedVariable, reportUndefinedVariable]
    except Exception as ae:
        if str(ae).find("[TABLE_OR_VIEW_NOT_FOUND]") != -1:
            logger.warning(f"[drug] mapping table {ENTITY_MAPPING_TABLE} not found. Using empty DataFrame.")
        else:
            raise ae  # noqa: TRY201

    df_drug = (
        _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.drug")
        .alias("drug")  # pyright: ignore[reportArgumentType]
        .join(
            df_entity.alias("entity"),
            on=(
                F.lower(F.col("drug.drugname")).eqNullSafe(F.lower(F.col("entity.drugname")))
                & F.lower(F.col("drug.prod_ai")).eqNullSafe(F.lower(F.col("entity.prod_ai")))
            ),
            how="left",
        )
    ).join(
        df_mapped_entity.alias("mapped"),
        on=F.lower(F.col("entity.entity")).eqNullSafe(F.lower(F.col("mapped.entity"))),
        how="left",
    )

    logger.info(f"[entity_atc_code|out] => {df_drug}")
    return df_drug


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.demo")
def silver_demo() -> DataFrame:
    """Read processed DEMO data and return silver table with deleted cases removed."""
    return _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.demo")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.outc")
def silver_outc() -> DataFrame:
    """Read processed OUTC data and return silver table with deleted cases removed."""
    return _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.outc")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.rpsr")
def silver_rpsr() -> DataFrame:
    """Read processed RPSR data and return silver table with deleted cases removed."""
    return _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.rpsr")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.ther")
def silver_ther() -> DataFrame:
    """Read processed THER data and return silver table with deleted cases removed."""
    return _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.ther")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.reac", partition_cols=["period"])
def silver_reac() -> DataFrame:
    """Read processed REAC data and return silver table with deleted cases removed."""
    logger.info("[silver_reac|in]")
    df = _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.reac")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]
    df_meddra_pt = spark.read.table(f"`{MEDDRA_TABLES_SCHEMA}`.meddra_preferred_terms")  # pyright: ignore[reportUndefinedVariable]
    df_meddra_llt_former_pt = spark.read.table(f"`{MEDDRA_TABLES_SCHEMA}`.meddra_llt_former_pt")  # pyright: ignore[reportUndefinedVariable]

    result: DataFrame = MedDRAMapping().process(
        {
            "dataset": df,
            "table": "reac",
            "meddra_datasets": {"preferred_terms": df_meddra_pt, "llt_former_pt": df_meddra_llt_former_pt},
        }
    )
    logger.info(f"[silver_reac|out] => {result}")
    return result


@dp.table(name=f"`{SILVER_CATALOG}`.{SCHEMA}.indi", partition_cols=["period"])
def silver_indi() -> DataFrame:
    """Read processed INDI data and return silver table with deleted cases removed."""
    logger.info("[silver_indi|in]")
    df = _remove_deleted_cases(spark, f"`{BRONZE_CATALOG}`.{SCHEMA}.indi")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]
    df_meddra_pt = spark.read.table(f"`{MEDDRA_TABLES_SCHEMA}`.meddra_preferred_terms")  # pyright: ignore[reportUndefinedVariable]
    df_meddra_llt_former_pt = spark.read.table(f"`{MEDDRA_TABLES_SCHEMA}`.meddra_llt_former_pt")  # pyright: ignore[reportUndefinedVariable]

    result: DataFrame = MedDRAMapping().process(
        {
            "dataset": df,
            "table": "indi",
            "meddra_datasets": {"preferred_terms": df_meddra_pt, "llt_former_pt": df_meddra_llt_former_pt},
        }
    )
    logger.info(f"[silver_indi|out] => {result}")
    return result
