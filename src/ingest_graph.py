from typing import List, TypedDict
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from .config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME


class IngestState(TypedDict):
    url: str | None
    file_path: str | None
    source_type: str
    source_name: str
    docs: List[Document]
    chunks: List[Document]


def load_source(state: IngestState) -> IngestState:
    if state.get("url"):
        loader = WebBaseLoader(state["url"])
        docs = loader.load()
        source_type, source_name = "website", state["url"]
    elif state.get("file_path"):
        if state["file_path"].lower().endswith(".pdf"):
            loader = PyPDFLoader(state["file_path"])
        else:
            loader = UnstructuredFileLoader(state["file_path"])
        docs = loader.load()
        source_type, source_name = "document", os.path.basename(state["file_path"])
    else:
        raise ValueError("Must provide url or file_path")

    for doc in docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["source_name"] = source_name

    return {
        **state,
        "docs": docs,
        "source_type": source_type,
        "source_name": source_name
    }


def split_docs(state: IngestState) -> IngestState:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(state["docs"])
    for chunk in chunks:
        chunk.metadata["source_type"] = state["source_type"]
        chunk.metadata["source_name"] = state["source_name"]
    
    return {
        **state,
        "chunks": chunks
    }


def store_in_pinecone(state: IngestState) -> IngestState:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    vectorstore = Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    vectorstore.add_documents(state["chunks"])
    return state



graph = StateGraph(IngestState)
graph.add_node("load_source", load_source)
graph.add_node("split_docs", split_docs)
graph.add_node("store_in_pinecone", store_in_pinecone)

graph.add_edge(START, "load_source")
graph.add_edge("load_source", "split_docs")
graph.add_edge("split_docs", "store_in_pinecone")
graph.add_edge("store_in_pinecone", END)

# Expose to LangGraph platform
graph_ingest = graph.compile()