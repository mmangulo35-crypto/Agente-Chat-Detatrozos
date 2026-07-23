
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    
    carpeta = Path(sys.executable).parent
else:
    
  carpeta = Path(__file__).parent

documentos = []

for archivo in carpeta.glob("*.pdf"):
    print(f"Cargando: {archivo.name}")
    loader = PyPDFLoader(str(archivo))
    documentos.extend(loader.load())
    print(f"Documentos cargados: {len(documentos)}")

if len(documentos) == 0:
    raise Exception("No se encontró ningún PDF en la carpeta.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
fragmentos = text_splitter.split_documents(documentos)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(fragmentos, embeddings)
llm = ChatOllama (model="llama3.2", temperature=0.2)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
system_prompt = (
    "Eres un agente de la tienda virtual Detatrozos para responder preguntas que te hagan por al chat. "
    "Si no conoces la respuesta, di que no tienes esa información. "
    "\n\n"
    "{context}"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
while True:
    pregunta = input("\nEscribe tu pregunta (o escribe salir): ")

    if pregunta.lower() == "salir":
        break

    response = rag_chain.invoke({"input": pregunta})
    print("\nRespuesta:")
    print(response["answer"])
