import os
import io
import uuid
import base64
import time 
from base64 import b64decode
import numpy as np
from PIL import Image
import pydantic
from pydantic_ai import Agent
from pydantic_ai.models import openai
from pydantic_ai.providers import openai

from unstructured.partition.pdf import partition_pdf

from load_api import Settings
from agents import OpenAIAgentInit

settings = Settings()
agent = OpenAIAgentInit()

table_agent = agent.create_table_agent()
image_agent = agent.create_img_agent()
chat_agent = agent.create_chat_agent()

# load the pdf file to drive
# split the file to text, table and images
def doc_partition(path,file_name):
  raw_pdf_elements = partition_pdf(
    filename=path + file_name,
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path)

  return raw_pdf_elements
path = "/content/"
file_name = "wildfire_stats.pdf"
raw_pdf_elements = doc_partition(path,file_name)


# appending texts and tables from the pdf file
def data_category(raw_pdf_elements): # we may use decorator here
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
           tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
           texts.append(str(element))
    data_category = [texts,tables]
    return data_category
texts = data_category(raw_pdf_elements)[0]
tables = data_category(raw_pdf_elements)[1]

# function to take tables as input and then summarize them
def tables_summarize(data_category):
    prompt_text = """You are an assistant tasked with summarizing tables. \
                    Give a concise summary of the table. Table chunk: {element} """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    

    return table_summaries
table_summaries = tables_summarize(data_category)
text_summaries = texts

