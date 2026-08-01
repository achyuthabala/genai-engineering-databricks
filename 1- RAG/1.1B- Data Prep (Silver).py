# Databricks notebook source
# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

bronze_df = spark.read.table("bronze_docs_raw")
display(bronze_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH parsed_docs AS (
# MAGIC   SELECT path,
# MAGIC         ai_parse_document(content, map('version', '2.0')) as parsed_content
# MAGIC   FROM bronze_docs_raw
# MAGIC )
# MAGIC SELECT path,
# MAGIC       parsed_content,
# MAGIC       ai_query('databricks-gpt-oss-20b',
# MAGIC                CONCAT('Given a JSON object representing a parsed document (with pages, elements, and metadata), convert the content into clean, readable markdown. Use "== page ==" to separate each page without adding page number. Preserve important structure such as pages, headers, tables, and captions. Do not include any JSON or code blocks in the output—just the clean markdown text.
# MAGIC                JSON:', CAST(parsed_content AS STRING))
# MAGIC               )
# MAGIC FROM parsed_docs

# COMMAND ----------

from pyspark.sql.functions import expr

endpoint = "databricks-gpt-oss-20b"

prompt_prefix = '''
You are a helpful assistant. Given a JSON object representing a parsed document (with pages, elements, and metadata), convert the content into clean, readable markdown. Use "== page ==" to separate each page without adding page number. Preserve important structure such as pages, headers, tables, and captions. Do not include any JSON or code blocks in the output—just the clean markdown text.

JSON:

'''

parsed_df = (bronze_df.withColumn("parsed_content",
                                  expr(f"""ai_parse_document(content, map("version", "2.0",
                                                                          "imageOutputPath", "{course.dataset_volume}/imgs")
                                                            )"""))
                      .withColumn("plain_text",
                                  expr(f"""ai_query('{endpoint}',
                                                    CONCAT('{prompt_prefix}', CAST(parsed_content AS STRING)),
                                                    responseFormat => '{{"type":"text"}}'
                                                  )"""))
                      .select("path", "parsed_content", "plain_text")
            )

display(parsed_df)

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath('../Includes'))
from document_renderer import render_ai_parse_output, render_ai_parse_output_interactive

sample = parsed_df.select("parsed_content").limit(1).collect()
doc = sample[0]["parsed_content"]
render_ai_parse_output(doc)

# COMMAND ----------

parsed_df.write.mode("overwrite").saveAsTable("silver_docs_parsed")