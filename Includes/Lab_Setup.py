# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import time

# COMMAND ----------

def path_exists(path):
  try:
    dbutils.fs.ls(path)
    return True
  except Exception as e:
    msg = str(e)
    if ("com.databricks.sql.io.CloudFileNotFoundException" in msg
        or "java.io.FileNotFoundException" in msg):
      return False
    else:
      raise

# COMMAND ----------

class CourseSetup:
    def __init__(self, uri, catalog_name, schema_name):
        self.uri = uri
        self.catalog = catalog_name
        self.schema = schema_name
    
    def download_dataset(self):
        source = self.uri
        target = self.dataset_volume

        files = dbutils.fs.ls(source)

        for f in files:
            source_path = f"{source}/{f.name}"
            target_path = f"{target}/{f.name}"
            if not path_exists(target_path):
                print(f"Copying {f.name} ...")
                dbutils.fs.cp(source_path, target_path, True)
    
    
    def create_database(self):
        spark.sql(f"USE CATALOG {self.catalog}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        spark.sql(f"USE SCHEMA {self.schema}")

        print(f"Catalog name: {self.catalog}")
        print(f"Schema name: {self.schema}")

        self.__configure_directories()
    
    
    def clean_up(self):
        print("Dropping Database, Dataset, Cache, and Checkpoints ...")
        spark.sql(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
        print("Done")


    def __configure_directories(self):
        dataset_volume_name = "dataset"
        checkpoint_volume_name = "_checkpoint"
        cache_volume_name = "_cache"

        volume_root = f"/Volumes/{self.catalog}/{self.schema}"
        self.dataset_volume = f"{volume_root}/{dataset_volume_name}"
        self.checkpoint_volume = f"{volume_root}/{checkpoint_volume_name}"
        self.cache_volume = f"{volume_root}/{cache_volume_name}"
        
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {dataset_volume_name}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {checkpoint_volume_name}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cache_volume_name}")
    
    def __get_index(self, dir):
        try:
            files = dbutils.fs.ls(dir)
            file = max(f.name for f in files if f.name.endswith('.json'))
            index = int(file.rsplit('.', maxsplit=1)[0])
        except:
            index = 0
        return index+1
    

    def create_agent_tables(self):
        phones_table = "smartphones"
        inventory_table = "inventory_stock"

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {phones_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/smartphones.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{phones_table} table is ready.")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {inventory_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/inventory_stock.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{inventory_table} table is ready.")

    def create_batch_tables(self):
        tickets_table = "customer_support_tickets"

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {tickets_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/support_tickets.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{tickets_table} table is ready.")

# COMMAND ----------

data_source_uri = "s3://dalhussein-courses/GenAI-Eng/datasets/dphone/v1/"
schema_name = "genai_course"

catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
course = CourseSetup(data_source_uri, catalog_name, schema_name)

course.create_database()
course.download_dataset()