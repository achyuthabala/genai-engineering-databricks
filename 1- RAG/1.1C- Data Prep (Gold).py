# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain_text_splitters
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

silver_df = spark.read.table("silver_docs_parsed")
display(silver_df)

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n== page ==\n", "== page ==", "\n\n", "\n", " ", ""]
)

def chunk_text(text):
    chunks = []
    if isinstance(text, str) and text.strip():
        for c in splitter.split_text(text):
            if c and c.strip():
                chunks.append(c)
    return chunks

# COMMAND ----------

doc = """DerarTel MOBILE TECHNOLOGY ## Product Specifications: dPhone D1 Status: Internal/Confidential ### 1. Executive Summary The dPhone D1 is engineered to redefine the premium smartphone market by bridging the gap between high-performance computing and minimalist design. The D1 prioritizes Open Ecosystem integration, Adaptive Hardware, and Quantum-Safe Encryption. > dPhone D1 > Fluid OS > March 28, 2028 > dPhone D1 | Quantum-Safe Encryption > Invisible under-display tech > dPhone D1 | Matte 95% recycled titanium > Apex Display > Apex Secondary > Sealed dual-purpose microSD slot and Hydro-Guard Seal **Figure 1.** dPhone D1 product image showcasing the full design and display features == page == ### 2. Key Value Propositions The “Fluid” OS: A proprietary operating system built on a microkernel architecture that allows for 100% app compatibility with both Android and iOS environments. Sustainability: Built with a 93% recycled titanium chassis and a user‑replaceable battery. | Feature | Specification | |---------|---------------| | Processor | Helios G1M Neural Chip (5nm architecture) | | Display | 6.5" Super‑AMOLED “Infinity” Display | | Storage | 512GB | | Refresh Rate | Dynamic 1Hz to 144Hz | | Battery | 5,500 mAh with 60W Rapid Charging | | Security | Biometric “Vein‑Scan” (Under‑display) | | Connectivity | 6G Ready & Satellite SOS | ### 3. Environmental Resistance To ensure maximum durability, the dPhone D1 features industry‑leading waterproof protection: IP69 Rating: The highest rating for ingress protection, certifying the dPhone D1 is completely dust‑tight and capable of withstanding high‑pressure, high‑temperature water jets. Hydro‑Guard Internal Seal: Proprietary nanocoating protects internal components. == page == ### 4. Performance Benchmarks (Scale 0‑100) AI Processing: The dPhone D1 scores a 98, leveraging the Helios G1 Neural Chip, compared to the competitor's 85. - Battery Life: Optimized power management gives the D1 a score of 95, outlasting the 88 of the rival. Thermal Efficiency: With its advanced liquid‑cooling and titanium chassis, the D1 maintains 92% efficiency under load. Signal Stability: The integrated 6G‑ready modem provides a 94 score for connection reliability in weak signal areas. > dPhone D1 Performance Benchmarks > Performance Score (0‑100) > AI Processing > Battery Life > Thermal Efficiency > Signal Stability **Figure 2.** Performance comparison between the iPhone DI and the current industry leading flagship. The dPhone D1 outperforms its competitor across key performance metrics, excelling in AI processing with a 98/100 score thanks to the Helios G1 Neural Chip. Its optimized battery and thermal management deliver long‑lasting use and stable operation under heavy load. Additionally, the 6G‑ready modem ensures reliable connectivity even in weak signal areas. - End of Document -"""
chunks = chunk_text(doc)
print(f'Number of chunks: {len(chunks)}')
chunks

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf, explode
import pandas as pd

@pandas_udf("array<string>")
def get_chunks(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for s in batch_iter:
        yield s.apply(chunk_text)

df_chunks = (silver_df.withColumn("chunk", explode(get_chunks("plain_text")))
                                        .select("path", "chunk")

            )

display(df_chunks)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df_chunks = df_chunks.withColumn("id", monotonically_increasing_id())

df_chunks.write.mode("overwrite").saveAsTable("gold_docs_chunked")