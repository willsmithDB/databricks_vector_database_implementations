# Databricks notebook source
# UC Variables 
catalog_name = "vd_catalog"
schema_name = "vd_implementations"

# TODO LOCAL PERSIST DIR to your preferred path
LOCAL_PERSIST_DIR = "/dbfs/FileStore/{schema_name}/faiss_index"
PROMPT_TEMPLATE = "Given the {context} answer what is the {question}"


embedding_endpoint = "databricks-bge-large-en"
llm_endpoint = "databricks-dbrx-instruct"
