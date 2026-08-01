# Databricks notebook source
# MAGIC %pip install --quiet --upgrade evaluate tiktoken transformers torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dphone_inference_payload

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, element_at

def unpack_requests(requests_raw: DataFrame, 
                    input_json_path: str, 
                    input_schema: str, 
                    output_json_path: str, 
                    output_schema: str) -> DataFrame:

    requests_success = requests_raw.filter(col("status_code") == "200")

    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_json_path}"), input_schema))
        .withColumn("response", F.from_json(F.expr(f"response:{output_json_path}"), output_schema))
    )

    last_messages = (requests_unpacked.withColumn("input", F.element_at(F.col("request"), -1))
                                     .withColumn("output", F.element_at(F.col("response"), -1)[0])
                                     .drop("request", "response")
    )
    
    return last_messages

# COMMAND ----------

# Input query format: {"input": [{"content": "User question? ..."}]}
INPUT_PATH = "input[*].content"
INPUT_SCHEMA = "array<string>"

# Response format: {"output": [{"type": "reasoning", "summary": [...],},
#                              {"type": "function_call", "name": "...",},
#                              {"content": [{"text": "Model response ..."}]}]}
OUTPUT_PATH = "output[*].content[*].text"
OUPUT_SCHEMA = "array<array<string>>"

# COMMAND ----------

inference_table_name = f"{course.catalog}.{course.schema}.dphone_inference_payload"

sample_df = spark.table(inference_table_name).where('status_code == 200').limit(10)
unpacked_sample_df = unpack_requests(
    sample_df,
    INPUT_PATH,
    INPUT_SCHEMA,
    OUTPUT_PATH,
    OUPUT_SCHEMA
)

display(unpacked_sample_df)

# COMMAND ----------

import os
import tiktoken, evaluate
import pandas as pd
from pyspark.sql.functions import pandas_udf

cache_dir = f"{course.cache_volume}/huggingface"

@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))
 
@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  os.environ['HF_HOME'] = cache_dir
  import evaluate

  toxicity = evaluate.load("toxicity", module_type="measurement")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def compute_metrics(requests_df: DataFrame, column_to_measure = ["input", "output"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"token_count_{column_name}", compute_num_tokens(col(column_name)))
                 .withColumn(f"toxicity_{column_name}", compute_toxicity(col(column_name)))
    )
  return requests_df

display(compute_metrics(unpacked_sample_df))

# COMMAND ----------

from delta.tables import DeltaTable

def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    (
      DeltaTable.createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .execute()
    )

# COMMAND ----------

checkpoint_location = f"{course.cache_volume}/dphone_processed_inferences"

requests_raw_df = spark.readStream.table(inference_table_name)
requests_processed_df = unpack_requests(
    requests_raw_df,
    INPUT_PATH,
    INPUT_SCHEMA,
    OUTPUT_PATH,
    OUPUT_SCHEMA
)

requests_processed_df = requests_processed_df.drop("databricks_request_id", "request_date", "client_request_id", "status_code", "sampling_fraction", "logging_error_codes" , "requester")

requests_with_metrics_df = compute_metrics(requests_processed_df)

processed_table_name = f"{course.catalog}.{course.schema}.dphone_processed_inferences"
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics_df)

# COMMAND ----------

(requests_with_metrics_df.writeStream
                      .trigger(availableNow=True)
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name)
                      .awaitTermination())

display(spark.table(processed_table_name))