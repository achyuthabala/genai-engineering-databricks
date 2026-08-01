# Databricks notebook source
# MAGIC %pip install --quiet --upgrade unitycatalog-ai
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup_02

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM smartphones

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM inventory_stock

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_latest_phone()
# MAGIC RETURNS TABLE(phone_id INT, model STRING, color STRING, price INT)
# MAGIC COMMENT 'Returns the latest phone, ordered by release_date descending. IMPORTANT: This function accepts ZERO parameters. Do not pass any arguments.'
# MAGIC RETURN
# MAGIC   SELECT phone_id, model, color, price
# MAGIC   FROM smartphones
# MAGIC   ORDER BY release_date DESC
# MAGIC   LIMIT 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_latest_phone()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION check_inventory(phone_id INT COMMENT 'ID of the phone to check (integer)')
# MAGIC RETURNS TABLE (
# MAGIC   item_id INT,
# MAGIC   quantity INT
# MAGIC )
# MAGIC COMMENT 'Checks the stock quantity for a given phone in the inventory.'
# MAGIC RETURN
# MAGIC   SELECT item_id, quantity
# MAGIC   FROM inventory_stock
# MAGIC   WHERE item_id = phone_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM check_inventory(4)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_current_user()
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the current session user, used for identifying the active user. IMPORTANT: This function accepts ZERO parameters. Do not pass any arguments.'
# MAGIC RETURN current_user();

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT get_current_user()

# COMMAND ----------

def generate_checkout_url(user_id: str, phone_id: int, valid_quantity: int) -> str:
    """
    Returns the checkout URL to fulfill the user's order for a phone with a given quantity.
    The URL is valid for 30 minutes.

    Args:
        user_id (str): Identifier of the current user/session.
        phone_id (int): ID of the phone to buy.
        valid_quantity (int): the requested quantity, validated against current stock

    Returns:
        str: Checkout URL string.
    """
    from datetime import datetime, timedelta
    from urllib.parse import urlencode

    expiry_time = datetime.now() + timedelta(minutes=30)
    params = {
        "i": phone_id,
        "q": valid_quantity,
        "u": user_id,
        "t": expiry_time
    }
    
    # Return checkout URL
    checkout_url = f"https://www.derar.cloud/checkout?{urlencode(params)}"
    return checkout_url

# COMMAND ----------

generate_checkout_url("derar@derar.cloud", 4, 2)

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

python_tool_uc_info = client.create_python_function(func=generate_checkout_url,
                                                    catalog=course.catalog,
                                                    schema=course.schema,
                                                    replace=True)