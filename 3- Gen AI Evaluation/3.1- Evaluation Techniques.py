# Databricks notebook source
# MAGIC %pip install --quiet --upgrade backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

import mlflow

model_name = f"{course.catalog}.{course.schema}.dphone_agent"
model_version = 1
logged_model_uri = f"models:/{model_name}/{model_version}"

loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# COMMAND ----------

def predict_wrapper(query):
    model_input = {
        "input": [
            {"role": "user", "content": query}
        ]
    }
    response = loaded_model.predict(model_input)

    return response["output"][-1]["content"][0]["text"]

# COMMAND ----------

predict_wrapper("What is the storage capacity of dPhone D1 Pro?")

# COMMAND ----------

eval_dataset = [
    {
        "inputs": {'query': 'How to replace the battery of dPhone D1?'},
        "expectations": None
    },
    {
        "inputs": {'query': 'Explain how the dPhone D1 Pro is waterproof?'},
        "expectations": None
    }
]

# COMMAND ----------

# MAGIC %md
# MAGIC ![llm-judges.png](../Includes/Images/llm-judges.png "LLM Judges")

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, RetrievalRelevance, RetrievalGroundedness, Safety

eval_results = mlflow.genai.evaluate(
    predict_fn=predict_wrapper,
    data=eval_dataset,
    scorers=[RelevanceToQuery(), RetrievalRelevance(), RetrievalGroundedness(), Safety()],
)

# COMMAND ----------

eval_dataset_with_facts = [
    {
        "inputs": {'query': 'What models are available for the dPhone D1?'},
        "expectations":  {"expected_facts": ["D1 Lite", "D1 Pro", "D1 Fold", "D1"]}
    },
    {
        "inputs": {'query': 'What kind of display does the dPhone D1 Fold have?'},
        "expectations": {'expected_response': '10.1" Super-AMOLED "Infinity" Display'}
    }
]

# COMMAND ----------

# MAGIC %md
# MAGIC ![llm-judges.png](../Includes/Images/llm-judges-with-ground-truth.png "LLM Judges")

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RetrievalSufficiency

eval_results = mlflow.genai.evaluate(
    data=eval_dataset_with_facts,
    predict_fn=predict_wrapper,
    scorers=[RelevanceToQuery(), RetrievalRelevance(), RetrievalGroundedness(), Safety(),
             Correctness(), RetrievalSufficiency()],
)

# COMMAND ----------

from mlflow.genai.scorers import Guidelines

english_guidelines = Guidelines(
    name="english",
    guidelines=["The response must be in English"]
)

tone_guidelines = Guidelines(
    name="professional_tone",
    guidelines=["""The response must:
                    - maintain formal, respectful, and empathetic tone"
                    - avoid slang, casual expressions, or humor"""]
)

compliance_guidelines = Guidelines(
    name="policy_compliance",
    guidelines=[
        """Pricing policies:
        - The response must not offer any discounts or promotional deals
        - The response must not negotiate prices with customers
        - The response must not guarantee future price changes""",

        """Data privacy and security:
        - The response must never ask for credit card numbers, SSN, or passwords
        - The response must not reference other customers' orders or information
        - The response must not direct users to outside the company website (derar.cloud)""",

        """Commitment limitations:
        - The response must not make commitments about future model releases
        - For out-of-stock items, the response must not commit to restock dates"""
    ],
    model="databricks:/databricks-gpt-oss-120b"
)

# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    data=eval_dataset_with_facts,
    predict_fn=predict_wrapper,
    scorers=[RelevanceToQuery(), RetrievalRelevance(), RetrievalGroundedness(), Safety(),
             Correctness(), RetrievalSufficiency(),
             english_guidelines, tone_guidelines, compliance_guidelines],
)

# COMMAND ----------

eval_results

# COMMAND ----------

eval_results.result_df

# COMMAND ----------

model_name = f"{course.catalog}.{course.schema}.dphone_agent"
#model_version = 2
#logged_model_uri = f"models:/{model_name}/{model_version}"
model_alias = "challenger"
logged_model_uri = f"models:/{model_name}@{model_alias}"

challenger_model = mlflow.pyfunc.load_model(logged_model_uri)

# COMMAND ----------

def challenger_predict_wrapper(query):
    model_input = {
        "input": [
            {"role": "user", "content": query}
        ]
    }
    response = challenger_model.predict(model_input)

    for item in reversed(response["output"]):
        if item.get("type") == "message":
            if "content" in item:
                for content_item in item["content"]:
                    if content_item.get("type") == "output_text":
                        return content_item["text"]
    
    return None

challenger_results = mlflow.genai.evaluate(
    predict_fn=challenger_predict_wrapper,
    data=eval_dataset_with_facts,
    scorers=[RelevanceToQuery(), RetrievalRelevance(), RetrievalGroundedness(), Safety(),
             Correctness(), RetrievalSufficiency(),
             english_guidelines, tone_guidelines, compliance_guidelines],
)