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

#doc_batches = [random.sample(docs,1), random.sample(docs,1)]
# Assuming you have 'docs' defined as a list of documents
total_rows = len(docs)
batch_size = max(total_rows // 4, 1000)  # Adjust the batch size as needed

# Split 'docs' into three equal-sized batches
doc_batches = [docs[i:i + batch_size] for i in range(0, total_rows, batch_size)]

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
    time.sleep(60)

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

template = """Use the following pieces of course syllabus details to construct or aggregate the course syllabus at the end.
Give course syllabus week by week for up to 4 weeks. Give syllabus for each week in atleast three points, describing the topics to be covered in that week.
If you do not find relevant information from the dataset, please create one.
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": compression_retriever_reordered | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
ans = rag_chain.invoke("Give course syllabus for music and Machine learning?")
print(ans)
# @app.route('/createcoursework', methods=['POST'])
# def ask_question():
#     data = request.json
#     context = retriever | format_docs
#     question = data.get('question', '')
#     result = rag_chain.invoke({"context": context, "question": question})
#     return jsonify({"answer": result})

# if __name__ == '__main__':
#     app.run(debug=True)
#     print("App Running.....")
