import os
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------------------------
# STATE
# ---------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]   # Conversation history
    website_url: str
    retrieved_docs: List[Document]

# ---------------------------
# COMPONENTS
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# Persistent Chroma DB
vectorstore = Chroma(
    collection_name="website_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ---------------------------
# SCRAPE WEBSITE
# ---------------------------
def scrape_website(state: State) -> State:
    url = state.get("website_url")
    if not url:
        return state

    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split content
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Add to vectorstore
    vectorstore.add_documents(split_docs)

    return state

# ---------------------------
# RETRIEVE FROM VECTORSTORE
# ---------------------------
def retrieve_docs(state: State) -> State:
    if not state["messages"]:
        return state

    query = state["messages"][-1].content  # Last user message
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    state["retrieved_docs"] = docs
    return state

# ---------------------------
# RAG CHATBOT
# ---------------------------
def rag_chatbot(state: State):
    context = "\n\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
    user_message = state["messages"][-1].content if state["messages"] else ""

    prompt = f"""You are a helpful assistant.
Use the following website context if relevant:

{context}

User question: {user_message}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}  # appended to history

# ---------------------------
# FALLBACK CHATBOT
# ---------------------------
def fallback_chatbot(state: State):
    user_message = state["messages"][-1].content if state["messages"] else ""
    response = llm.invoke([HumanMessage(content=user_message)])
    return {"messages": [response]}  # appended to history

# ---------------------------
# CONDITIONAL EDGE
# ---------------------------
def decide_next_node(state: State) -> str:
    if state.get("retrieved_docs"):  # If docs found
        return "rag_chatbot"
    return "fallback_chatbot"

# ---------------------------
# BUILD GRAPH
# ---------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("scrape_website", scrape_website)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("rag_chatbot", rag_chatbot)
graph_builder.add_node("fallback_chatbot", fallback_chatbot)

graph_builder.add_edge(START, "scrape_website")
graph_builder.add_edge("scrape_website", "retrieve_docs")
graph_builder.add_conditional_edges("retrieve_docs", decide_next_node)
graph_builder.add_edge("rag_chatbot", END)
graph_builder.add_edge("fallback_chatbot", END)

graph = graph_builder.compile()
