# Databricks notebook source
# MAGIC %run ../Includes/Lab_Setup_04

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput
from databricks.sdk import WorkspaceClient

serving_endpoint_name = f"dphone_endpoint"
model_fullname = f"{course.catalog}.{course.schema}.dphone_agent"
latest_model_version = 2

endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_fullname,
            "model_version": latest_model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
            "environment_vars": {},
        },
    ]
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

w = WorkspaceClient()
w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

# COMMAND ----------

customer_requests_table = f"{course.catalog}.{course.schema}.customer_support_tickets"
customer_requests_df = spark.read.table(customer_requests_table).filter("status = 'open'").limit(2)
display(customer_requests_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT description,
# MAGIC        ai_query(endpoint =>"dphone_endpoint", request => description) response
# MAGIC FROM customer_support_tickets
# MAGIC Where status = 'open'