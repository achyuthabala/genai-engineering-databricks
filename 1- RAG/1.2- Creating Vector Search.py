# Databricks notebook source
# MAGIC %pip install --quiet --upgrade databricks-vectorsearch flashrank
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)
vs_endpoint_name = "vs_endpoint"

if vs_client.endpoint_exists(vs_endpoint_name):
  vs_client.wait_for_endpoint(vs_endpoint_name)
else:
  vs_client.create_endpoint_and_wait(vs_endpoint_name, endpoint_type='STANDARD')

# COMMAND ----------

source_table_fullname = f"{course.catalog}.{course.schema}.gold_docs_chunked"

spark.sql(f"ALTER TABLE {source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"

if not vs_client.index_exists(vs_endpoint_name, vs_index_fullname):
  vs_index = vs_client.create_delta_sync_index_and_wait(
    source_table_name=source_table_fullname,
    index_name=vs_index_fullname, 
    primary_key="id",
    embedding_source_column="chunk",
    embedding_model_endpoint_name="databricks-gte-large-en",
    endpoint_name=vs_endpoint_name,
    pipeline_type="TRIGGERED", # TRIGGERED or STREAMING
  )
else:
  vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)
  vs_index.wait_until_ready(wait_for_updates=True)

# COMMAND ----------

query_1 = "Explain the features of the dPhone D1 Lite"
results = vs_index.similarity_search(
    query_text=query_1,
    columns=["path", "chunk"],
    num_results=3,
)

passages = results.get("result", {}).get("data_array", [])

display(passages)

# COMMAND ----------

query_2 = "6G broadband"
results_fulltext = vs_index.similarity_search(
    query_text=query_2,
    columns=["path", "chunk"],
    query_type="FULL_TEXT",
    num_results=3
)

passages_fulltext = results_fulltext.get("result", {}).get("data_array", [])

display(passages_fulltext)

# COMMAND ----------

query_3 = "Which products have IP69?"
results_hybrid = vs_index.similarity_search(
    query_text=query_3,
    columns=["path", "chunk"],
    query_type="hybrid",
    num_results=5
)

passages_hybrid = results_hybrid.get("result", {}).get("data_array", [])

display(passages_hybrid)

# COMMAND ----------

query_4 = "How to replace the battery of dPhone D1?"
results_filtered = vs_index.similarity_search(
    query_text=query_4,
    columns=["path", "chunk"],
    filters={"path NOT" : f"dbfs:/Volumes/workspace_7474652544329313/genai_course/dataset/docs/08_User_Manual.pdf"},
    num_results=3
)

passages_filtered = results_filtered.get("result", {}).get("data_array", [])

display(passages_filtered)

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker

query_5 = "Explain the Halo camera in dPhone?"

results_reranked = vs_index.similarity_search(
    query_text=query_5,
    columns=["path", "chunk"],
    num_results=4,
    reranker=DatabricksReranker(columns_to_rerank=["chunk"])
)

reranked_passages = results_reranked.get("result", {}).get("data_array", [])

display(reranked_passages)

# COMMAND ----------

from flashrank import Ranker, RerankRequest

cache_dir = f"{course.cache_volume}/flashrank"
ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

docs = [
    {"file": doc[0], "text": doc[1]}
    for doc in results.get("result", {}).get("data_array", [])
]

rerankrequest = RerankRequest(query=query_1, passages=docs)
passages_flashrank = ranker.rerank(rerankrequest)

formatted_output = [
    {"file": doc["file"], "text": doc["text"], "score": float(doc["score"])} 
    for doc in passages_flashrank
]

display(formatted_output)