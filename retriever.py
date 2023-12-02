import bs4
from flask import Flask, request, jsonify
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Memory
import time
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import LongContextReorder
import pickle
import random
import os

os.environ["OPENAI_API_KEY"] = "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"

loader=CSVLoader(file_path="langChain-API/coursera__courses_dataset.csv")
docs = loader.load()
print(len(docs))

doc_batches = [random.sample(docs,1), random.sample(docs,1)]
# Assuming you have 'docs' defined as a list of documents
# total_rows = len(docs)
# batch_size = max(total_rows // 4, 1000)  # Adjust the batch size as needed

# # Split 'docs' into three equal-sized batches
# doc_batches = [docs[i:i + batch_size] for i in range(0, total_rows, batch_size)]

for i in doc_batches:
    print(len(i))

retriever_batches = []
for i, doc_batch in enumerate(doc_batches):
    print("preparing for batch ", i)
    vectorstore_batch = Chroma.from_documents(documents=doc_batch, embedding=OpenAIEmbeddings())
    retriever_batch = vectorstore_batch.as_retriever()
    print("Size of i = ", retriever_batch.__sizeof__())
    retriever_batches.append(retriever_batch)
    print("batch ", i, "trained, Sleeping for 1 min")
    #time.sleep(60)

final_retriever = MergerRetriever(retrievers=retriever_batches)
print(final_retriever.__sizeof__())
filter_embeddings = OpenAIEmbeddings()
filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
pipeline = DocumentCompressorPipeline(transformers=[filter])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=final_retriever
)

filter_ordered_cluster = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=10,
    num_closest=1,
)

filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=10,
    num_closest=1,
    sorted=True,
)

pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=final_retriever
)

filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=final_retriever
)