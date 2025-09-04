from typing import Optional
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from .config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME


class QueryState:
    query: str = ""
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    answer: str = ""


def query_pinecone(state: QueryState) -> QueryState:
    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Wrap index in LangChain vectorstore
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Optional metadata filters
    filters = {}
    if state.source_type:
        filters["source_type"] = state.source_type
    if state.source_name:
        filters["source_name"] = state.source_name

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": filters if filters else {}}
    )
    llm = ChatOpenAI(model="gpt-4o-mini")

    docs = retriever.get_relevant_documents(state.query)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."

    prompt = f"Answer based on the ingested data:\n\nContext:\n{context}\n\nQuestion: {state.query}"
    response = llm.invoke(prompt)

    state.answer = response.content
    return state


graph = StateGraph(QueryState)
graph.add_node("query_pinecone", query_pinecone)
graph.add_edge(START, "query_pinecone")
graph.add_edge("query_pinecone", END)

# Expose to LangGraph platform
graph_query = graph.compile()
