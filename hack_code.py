# Databricks notebook source
# MAGIC %md
# MAGIC ## 1) Install Packages

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade databricks-vectorsearch langchain==0.2.0 langchain_community transformers databricks-cli mlflow mlflow[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##2) Load dataset, configure secrets & set up variables

# COMMAND ----------

import os
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("hackathon","token")

# COMMAND ----------

#Fill this out with the catalog where the data is stored and where all objects will be created (eg: tables, models etc)
catalog = "hackaton" #"hackathon" 

#Fill this out with the schema where the data is stored and where all objects will be created (eg: tables, models etc)
schema = "hackdata" #"default"

# COMMAND ----------

spark.sql(f"""USE CATALOG {catalog}""")
spark.sql(f"""USE DATABASE {schema}""")

# COMMAND ----------

# This is a reference cell that you do not need to use. You have already created tables using the UI. This is an example of how to do it through code:

# spark.read.csv('/Volumes/hackathon/default/engineering-notes/', header=True, inferSchema=True).write.saveAsTable('engineering_notes')

# COMMAND ----------

raw_data = spark.read.table("bbc_news") #<your table name>

# COMMAND ----------

#Replace the names below with the names of the endpoints for embedding model & chat model that would have been pre provisioned within the workspace. Replace the name for vector search endpoint & the name of the table to write to.
embedding_endpoint_name = "hackaton-embedding" #"bge_m3"
chat_endpoint_name = "hackaton-llama3-completion" #"meta_llama_3_8b_instruct"
vector_search_endpoint_name = "hackaton-vectorsearch"
table_name = "bbc_news_embeddings" #"engineering_notes_embeddings"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Explore your data

# COMMAND ----------

display(raw_data)

# COMMAND ----------

cleaned_data = raw_data.na.drop(subset=["description", "title"])

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

cleaned_data = cleaned_data.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

display(cleaned_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Chunk your data for vector search

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Splitting documentation pages into small chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC LLM models typically have a maximum input context length, and you won't be able to compute embeddings for very long texts.
# MAGIC In addition, the longer your context length is, the longer it will take for the model to provide a response.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC
# MAGIC - Split document into small chunks
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends on your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference
# MAGIC - Create multiple agents to evaluate each bigger document in parallel, and ask a final agent to craft your answer...
# MAGIC
# MAGIC
# MAGIC In this demo, we have big documentation articles, which are too long for the prompt to our model. We won't be able to use multiple documents as RAG context as they would exceed our max input size. Some recent studies also suggest that bigger window size isn't always better, as the LLMs seem to focus on the beginning and end of your prompt.
# MAGIC
# MAGIC In our case, we'll split these articles to ensure that each chunk is less than 500 tokens using LangChain. 
# MAGIC
# MAGIC #### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count tokens for a given sentence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer you'll be using here matches your model. We'll be using the `transformers` library to count llama2 tokens with its tokenizer. This will also keep our document token size below our embedding max size (1024).
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to manually review the chunks created and ensure that they make sense and contain relevant information.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from transformers import  AutoTokenizer
from langchain.text_splitter  import RecursiveCharacterTextSplitter 
from pyspark.sql import types as T


tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=500, chunk_overlap=50)

def get_chunks(c):
  chunks = []

  #Check if the chunks are small, if so add to them until they reach the midpoint of the max token size
  if len(tokenizer.encode(c)) <= 500/2:
    chunks.append(c)
  
  elif len(tokenizer.encode(c)) > 500/2:
    chunks.extend(text_splitter.split_text(c.strip()))

  return chunks


@pandas_udf(T.ArrayType(T.StringType()))
def get_chunks_udf(result: pd.Series) -> pd.Series:
  return result.apply(get_chunks)

# COMMAND ----------

from pyspark.sql import functions as F
data_chunked = cleaned_data.withColumn("chunks", get_chunks_udf("description")) \
                   .withColumn("chunk_exploded",F.explode("chunks")) \
                   .dropna(how = "any", subset = ['chunk_exploded']) \
                   .drop("chunks")

# COMMAND ----------

display(data_chunked)

# COMMAND ----------

# Delete this / adjust to the number of rows of data you want to work with
data_chunked = data_chunked.limit(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5) Create embeddings & set up vector search
# MAGIC
# MAGIC Databricks provides multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table 
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table
# MAGIC
# MAGIC In this demo, we will show you how to setup a **Self-managed Embeddings** index. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: right" width="800px">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1) Self managed embeddings

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

def compute_embeddings(batch):
  response = deploy_client.predict(endpoint=embedding_endpoint_name, inputs={"input": batch})
  return response['data'][0]['embedding']

# COMMAND ----------

# Edit the rows to match the names of your data columns

def apply_embedding(data_list):
  embedding_list = []
  for row in data_list:
    embedding_dict = {}
    embedding_dict['id'] = row['id']
    embedding_dict['title'] = row['title']
    embedding_dict['pubDate'] = row['pubDate']
    embedding_dict['guid'] = row['guid']
    embedding_dict['link'] = row['link']
    embedding_dict['description'] = row['description']
    embedding_dict['chunk_exploded'] = row['chunk_exploded']
    embedding_dict['embeddings'] = compute_embeddings(row['chunk_exploded'])
    embedding_list.append(embedding_dict)
  
  return embedding_list

# COMMAND ----------

data_to_embed_list = [row.asDict() for row in data_chunked.collect()]

# COMMAND ----------

embeddings = spark.createDataFrame(apply_embedding(data_to_embed_list))

# COMMAND ----------

embeddings.write.option("mergeSchema",True).mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

#This only needs to be configured once. Needed to use a table for vector search
spark.sql(f"""ALTER TABLE {catalog}.{schema}.{table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

vs_index_fullname = f"{catalog}.{schema}.{table_name}_pt_self_managed_embeddings"

def create_or_update_index():
  if len(vsc.list_indexes(vector_search_endpoint_name)) == 0 or vs_index_fullname not in [i['name'] for i in vsc.list_indexes(vector_search_endpoint_name)['vector_indexes']]:
    vsc.create_delta_sync_index(
      endpoint_name=vector_search_endpoint_name,
      index_name=vs_index_fullname,
      source_table_name=f"{catalog}.{schema}.{table_name}",
      primary_key="id",
      pipeline_type="TRIGGERED",
      embedding_dimension=1024,
      embedding_vector_column="embeddings"
    )
    print("Creating_index")
  else: 
    vsc.get_index(vector_search_endpoint_name, vs_index_fullname).sync()
    print("Updating_index")

create_or_update_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ###5.2) Databricks managed embeddings

# COMMAND ----------

#This only needs to be configured once. Needed to use a table for vector search
spark.sql(f"""ALTER TABLE {catalog}.{schema}.{table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

vs_index_fullname = f"{catalog}.{schema}.{table_name}_db_managed_embeddings"

def create_or_update_index():
  if len(vsc.list_indexes(vector_search_endpoint_name)) == 0 or vs_index_fullname not in [i['name'] for i in vsc.list_indexes(vector_search_endpoint_name)['vector_indexes']]:
    vsc.create_delta_sync_index(
      endpoint_name=vector_search_endpoint_name,
      index_name=vs_index_fullname,
      source_table_name=f"{catalog}.{schema}.{table_name}",
      primary_key="id",
      pipeline_type="TRIGGERED",
      embedding_source_column = "chunk_exploded",
      embedding_model_endpoint_name = embedding_endpoint_name
    )
    print("successfully created")

  else: 
    vsc.get_index(vector_search_endpoint_name, vs_index_fullname).sync()
    print("Index updated")

create_or_update_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ###6) Query vector search endpoint 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1 Query self managed index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#Note that we're querying a self managed embedding, so we need to convert to embeddings using our model endpoint
vs_index_fullname = f"{catalog}.{schema}.{table_name}_pt_self_managed_embeddings"
query = "Ukraine refugees"

#Get the embeddings that are most similar to the query
def search_similar_content(query):
  results = vsc.get_index(vector_search_endpoint_name, vs_index_fullname).similarity_search(
    #Compute the embeddings using our model endpoint through query_vector
    query_vector=compute_embeddings(query),
    columns=["id", "title", "description", "link"],
    num_results=5)

  #return [i[1] for i in results['result']['data_array']]
  return results

results_array = search_similar_content(query)
print(results_array)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2 Query Databricks managed index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#Note that we're querying a DB managed embedding, where databricks will manage the creation of embeddings from the question
vs_index_fullname = f"{catalog}.{schema}.{table_name}_db_managed_embeddings"
query = "Ukraine refugees"

#Get the embeddings that are most similar to the query
def search_similar_content(query):
  results = vsc.get_index(vector_search_endpoint_name, vs_index_fullname).similarity_search(
    query_text=query,
    columns=["id", "title", "description", "link"],
    num_results=5)
  
  return results
  #return [i[1] for i in results['result']['data_array']]

results_array = search_similar_content(query)
print(results_array)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7) Prompt engineering

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

response = deploy_client.predict(endpoint=chat_endpoint_name
                                 , inputs={"messages": [{"role":"user","content":f"Answer the question: {query} in 100 words or less using only the information available in: {results_array}. Include the link in your answer"}]})
print(response['choices'][0]['message']['content'])

# COMMAND ----------

# MAGIC %md
# MAGIC ###8) Build langchain retriever
# MAGIC Use langchain wrapper to stitch together the steps required:
# MAGIC * Create embeddings for question
# MAGIC * Call the vector search index & find similar indexes
# MAGIC * Calling a chat model and pass in the response into a prompt template
# MAGIC
# MAGIC Databricks Langchain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain.llms import Databricks

vs_index_fullname = f"{catalog}.{schema}.{table_name}_pt_self_managed_embeddings"
embedding_model = DatabricksEmbeddings(target_uri = "databricks",endpoint = embedding_endpoint_name)

#Wrap the retrieval of vectors in a retriever
def get_retriever(persist_dir: str = None):
  os.environ["DATABRICKS_HOST"] = host
  vsc = VectorSearchClient()
  vsc_index = vsc.get_index(
    endpoint_name = vector_search_endpoint_name, 
    index_name = vs_index_fullname
  )

  vectorstore = DatabricksVectorSearch(
    vsc_index, text_column = "description", embedding = embedding_model
  )
  return vectorstore.as_retriever()

vectorstore = get_retriever()

# COMMAND ----------

#This is to format the input to the model
def reformat_Func(prompt, **kwargs):
  # Note Single quotes is important!
  return { 
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 128
    }

chat_model = Databricks(allow_dangerous_deserialization=True,
                    transform_input_fn=reformat_Func,
                    endpoint_name=chat_endpoint_name)

chat_model._client.external_or_foundation = True

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(target_uri = "databricks",
                              endpoint = chat_endpoint_name)

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Create the prompt template and include the context and question
Template = """You are a question answering assistant. If the questions are not related to the data in the context, respond with 'I do not have enough information to answer'. Do not try to make up an answer. Answer the {question} in 100 words or less using the context below:
Context = {context}
Question = {question}"""


prompt = PromptTemplate(template = Template, input_variables = ['context','question'])

# COMMAND ----------

#Assemble the complete RAG chain
chain = RetrievalQA.from_chain_type(
  llm = chat_model,
  chain_type = "stuff",
  retriever = get_retriever(),
  chain_type_kwargs = {"prompt":prompt}

)

# COMMAND ----------

query = "Ukraine war"

question = {"query":query}
response = chain.invoke(question)

print(response['result'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9) Save your custom app as a UC model

# COMMAND ----------

import mlflow, langchain
mlflow.set_registry_uri("databricks-uc")
from mlflow.models import infer_signature

model_name = "bbc_news_rag"
model_name_3_level = f"{catalog}.{schema}.{model_name}"

with mlflow.start_run(run_name = "bbc_news_rag") as run:
  logged_model = mlflow.langchain.log_model(
    chain,
    loader_fn = get_retriever,
    artifact_path="chain",
    input_example = query,
    signature = infer_signature(query,response),
    registered_model_name = model_name,
    pip_requirements=[
          "langchain==0.2.0",
          "databricks-vectorsearch",
          "mlflow==2.13.0",
          "langchain_community"
    ]
  )


# COMMAND ----------

# MAGIC %md
# MAGIC ### 10) Create model serving endpoint

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
#Check for the latest version on UC to deploy
def get_latest_model_version(model_name_3_level):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name_3_level}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

print(get_latest_model_version(model_name_3_level))

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from datetime import timedelta
from mlflow import MlflowClient

client = MlflowClient()

w = WorkspaceClient()
serving_endpoint_name = 'bbc_news_rag'

endpoint_config = EndpointCoreConfigInput(
    served_entities =[
      ServedEntityInput(
      entity_name=f"{catalog}.{schema}.{model_name}",
      entity_version = get_latest_model_version(model_name_3_level),
      workload_size = "Small",
      scale_to_zero_enabled = True,
      workload_type = "CPU",
      environment_vars = {"DATABRICKS_TOKEN":"{{secrets/hackathon/token}}"}
      )
    ]
  )

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)

if existing_endpoint == None:
  w.serving_endpoints.create_and_wait(
    name = serving_endpoint_name,
    config = endpoint_config,
    timeout= timedelta(hours = 0, minutes = 40)
  )
  print("endpoint is being created")
else:
  w.serving_endpoints.update_config_and_wait(
    served_entities=endpoint_config.served_entities, 
    name=serving_endpoint_name,
    timeout= timedelta(hours = 0, minutes = 40)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11) Query endpoint

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "What are the typical resolution to pump failures"
response = deploy_client.predict(endpoint="engineering_notes_small", inputs={
  "dataframe_split": {
    "data": [
      [
        question
      ]
    ]
  }
})

# COMMAND ----------


