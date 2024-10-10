# Databricks notebook source
# DBTITLE 1,Version locked for verified compatibility
# MAGIC %pip install --force-reinstall mlflow==2.16.2 faiss-cpu==1.9.0 langchain==0.3.2 langchain_community==0.3.1 text-generation==0.7.0 sqlalchemy==2.0.35 langchain-databricks==0.1.0transformers==4.36.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config 

# COMMAND ----------


from langchain_databricks import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(
    endpoint=embedding_endpoint,
    # query_params={...},
    # document_params={...},
)

# COMMAND ----------

import transformers
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

LOCAL_PERSIST_DIR = LOCAL_PERSIST_DIR # Override here if needed

# Pull spam data to create documents from 
data = pd.read_csv(
    "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
    encoding="latin-1",
).sample(500)
data["id"] = range(0, len(data))

documents_list = [
    Document(
        page_content=r["v2"],
        metadata={"id": r["id"], "v1": r["v1"], "v2": r["v2"]},
    )
    for index, r in data.iterrows()
]

# COMMAND ----------

# DBTITLE 1,Instantiate faiss
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Create initial index
index = faiss.IndexFlatL2(len(embedding_model.embed_query("Initial faiss index")))

# Create the vector store 
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(documents=documents_list)

# COMMAND ----------

results = vector_store.similarity_search(
    "What are examples of Ham.",
    k=5,
    filter={},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save embeddings to local path

# COMMAND ----------

vector_store.save_local(LOCAL_PERSIST_DIR) # Can specific index_name

# COMMAND ----------

# DBTITLE 1,Test the retriver
# Example of using the retriever for a call
new_vector_store = FAISS.load_local(
    LOCAL_PERSIST_DIR, embedding_model, allow_dangerous_deserialization=True # required for pyfunc 
)

retriever = new_vector_store.as_retriever(search_kwargs={"k": 4})

query = "What is an example of ham?"
results = retriever.invoke(query)
print(results)

# COMMAND ----------

# DBTITLE 1,Alternative way of testing the vector store directly
results = new_vector_store.similarity_search(
    "What are examples of Ham.",
    k=5,
    filter={},
)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# COMMAND ----------

# DBTITLE 1,Verify that the number of items is correct
num_items = new_vector_store.index.ntotal

print(f"Number of items in the FAISS vector store: {num_items}") # Verify there are 500 items in the vector store

assert num_items == len(documents_list), "Number of items in the FAISS vector store does not match the number of items in the documents list"

# COMMAND ----------

# MAGIC %md
# MAGIC # Log with MLflow

# COMMAND ----------

import mlflow
import langchain
from langchain.vectorstores.faiss import FAISS
import sqlalchemy
import transformers
import cloudpickle
import torch
import pandas

artifacts = {"faissdb_path": LOCAL_PERSIST_DIR.replace("/dbfs", "dbfs:")}

conda_env = mlflow.pyfunc.get_default_conda_env()
packages = [
    "mlflow==2.16.2",
    "faiss-cpu==1.9.0",
    "langchain==0.3.2",
    "langchain_community==0.3.1",
    "text-generation==0.7.0",
    "sqlalchemy==2.0.35",
    "langchain-databricks==0.1.0",
    "pandas==2.2.3"
    f"transformers=={transformers.__version__}",
    f"torch=={torch.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
]

conda_env["dependencies"][-1]["pip"] += packages

# COMMAND ----------

class local_faiss_rag(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import mlflow
        import torch
        from langchain.vectorstores.faiss import FAISS
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_databricks import ChatDatabricks
        from langchain_databricks import DatabricksEmbeddings
        from langchain import hub
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain import hub
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
        prompt = hub.pull("rlm/rag-prompt")


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        embedding_model = DatabricksEmbeddings(
            endpoint=embedding_endpoint,
            # query_params={...},
            # document_params={...},
        )

        chat_model = ChatDatabricks(
            endpoint= llm_endpoint,
            temperature=0.1,
            max_tokens=256,
            # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
        )

        retriever_model = FAISS.load_local(
            folder_path=LOCAL_PERSIST_DIR, 
            embeddings=embedding_model,
            allow_dangerous_deserialization = True
            # "/dbfs/FileStore/faiss_index", embedding_model
        )

        prompt = hub.pull("rlm/rag-prompt")

        #  LCEL 
        self.qa_chain = (
            {
                "context": retriever_model.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt 
            | chat_model
            | StrOutputParser() 
        )

        #  High level Helper Methods if needed
        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat

        # combine_docs_chain = create_stuff_documents_chain(chat_model, retrieval_qa_chat_prompt)
        # self.qa_chain = create_retrieval_chain(retriever_model.as_retriever(), combine_docs_chain)

    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        """

        prompt = model_input["prompt"]
        input_query = prompt.iloc[0]

        print(type(input_query))
        print(f"Predict for the following prompt: {input_query} \n")

        # # temperature = params.get("temperature", 0.1) if params else 0.1
        # # max_tokens = params.get("max_tokens", 1000) if params else 1000

        result = self.qa_chain.invoke(input_query[0])

        return result

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
    ]
)
output_schema = Schema([ColSpec(DataType.string, "candidates")])

parameters = ParamSchema(
    [
        ParamSpec("temperature", DataType.float, np.float32(0.1), None),
        ParamSpec("max_tokens", DataType.integer, np.int32(1000), None),
    ]
)

signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)

# Define input example
input_example = pd.DataFrame({"prompt": ["What is an example of ham?"]})

# COMMAND ----------

import pandas as pd

# mlflow.set_experiment("/Shared/poa/llm-experiments")
mlflow.set_registry_uri("databricks")

with mlflow.start_run(run_name="local_faiss_db") as run:
    py_model = mlflow.pyfunc.log_model(
        artifact_path="models",
        python_model=local_faiss_rag(),
        conda_env=conda_env,
        artifacts=artifacts,
        input_example={"query": ["What is ham"]},
        registered_model_name="local_faiss_rag_db",
        signature = signature,
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(
    model_uri=py_model.model_uri
)

# COMMAND ----------

# params={"temperature": "true", "max_tokens": "256"}

inputs = [{"prompt": ["What examples in the context have the phrase \"buen tiempo\" from the vector database?"]}, {"prompt": ["What examples in the context have the phrase \"offer\" from the vector database?"]}] 

results = [loaded_model.predict(prompt) for prompt in inputs]

[print(f"Response: {item} \n") for item in results]

# COMMAND ----------

# MAGIC %md
# MAGIC # Local Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC import mlflow
# MAGIC import torch
# MAGIC from langchain.vectorstores.faiss import FAISS
# MAGIC from langchain.chains import RetrievalQA
# MAGIC from langchain.prompts import PromptTemplate
# MAGIC from langchain_databricks import ChatDatabricks
# MAGIC from langchain_databricks import DatabricksEmbeddings
# MAGIC
# MAGIC embedding_model = DatabricksEmbeddings(
# MAGIC     endpoint= embedding_endpoint,
# MAGIC     # query_params={...},
# MAGIC     # document_params={...},
# MAGIC )
# MAGIC
# MAGIC chat_model = ChatDatabricks(
# MAGIC     endpoint=llm_endpoint,
# MAGIC     temperature=0.1,
# MAGIC     max_tokens=256,
# MAGIC     # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
# MAGIC )
# MAGIC
# MAGIC retriever_model = FAISS.load_local(
# MAGIC     folder_path=LOCAL_PERSIST_DIR, 
# MAGIC     embeddings=embedding_model,
# MAGIC     allow_dangerous_deserialization = True
# MAGIC     # "/dbfs/FileStore/faiss_index", embedding_model
# MAGIC )
# MAGIC
# MAGIC prompt = PromptTemplate(
# MAGIC     template=PROMPT_TEMPLATE, input_variables=["context", "question"]
# MAGIC )
# MAGIC qa = RetrievalQA.from_chain_type(
# MAGIC # qa = RetrievalQA.from_chain_type(    
# MAGIC     llm=chat_model,
# MAGIC     chain_type="stuff",
# MAGIC     retriever=new_vector_store.as_retriever(k=5),
# MAGIC     return_source_documents=True
# MAGIC )
# MAGIC
# MAGIC qa.invoke("What examples in the context have the phrase \"offer\" from the vector database?")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Sample output:
# MAGIC ```
# MAGIC {'query': 'What examples in the context have the phrase "offer" from the vector database?',
# MAGIC  'result': '1. "EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per.."\n2. "U were outbid by simonwatson5120 on the Shinco DVD Plyr. 2 bid again, visit sms. ac/smsrewards 2 end bid notifications, reply END OUT"\n\nThe first example is an offer for a stock pick that has increased by over 300%. The second example is an offer to bid again on an online auction for a Shinco DVD player.',
# MAGIC  'source_documents': [Document(metadata={'id': 176, 'v1': 'ham', 'v2': 'How much r Ì_ willing to pay?'}, page_content='How much r Ì_ willing to pay?'),
# MAGIC   Document(metadata={'id': 205, 'v1': 'spam', 'v2': 'Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per..'}, page_content='Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per..'),
# MAGIC   Document(metadata={'id': 485, 'v1': 'spam', 'v2': 'U were outbid by simonwatson5120 on the Shinco DVD Plyr. 2 bid again, visit sms. ac/smsrewards 2 end bid notifications, reply END OUT'}, page_content='U were outbid by simonwatson5120 on the Shinco DVD Plyr. 2 bid again, visit sms. ac/smsrewards 2 end bid notifications, reply END OUT'),
# MAGIC   Document(metadata={'id': 203, 'v1': 'ham', 'v2': 'I need details about that online job.'}, page_content='I need details about that online job.')]}
# MAGIC   ```

# COMMAND ----------

# DBTITLE 1,Try with LCEL
# MAGIC %md
# MAGIC ```
# MAGIC from langchain import hub
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC
# MAGIC # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
# MAGIC prompt = hub.pull("rlm/rag-prompt")
# MAGIC
# MAGIC
# MAGIC def format_docs(docs):
# MAGIC     print("Formatting Documents \n")
# MAGIC     return "\n\n".join(doc.page_content for doc in docs)
# MAGIC
# MAGIC def print_prompt(prompt):
# MAGIC   print(f"Prompt: {prompt} \n")
# MAGIC   return prompt
# MAGIC
# MAGIC def print_response(response):
# MAGIC   print(f"Response: {response} \n")
# MAGIC   return response
# MAGIC
# MAGIC qa_chain = (
# MAGIC     {
# MAGIC         "context": new_vector_store.as_retriever() | format_docs,
# MAGIC         "question": RunnablePassthrough(),
# MAGIC     }
# MAGIC     | prompt | print_prompt
# MAGIC     | chat_model| print_response
# MAGIC     | StrOutputParser() | print_prompt
# MAGIC )
# MAGIC
# MAGIC qa.invoke("What examples in the context have the phrase \"offer\" from the vector database?")
# MAGIC ```

# COMMAND ----------


