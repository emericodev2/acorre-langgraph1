from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from config.settings import settings

class QAAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    
    def create_retriever(self, vector_store: Chroma, search_k: int = None):
        return vector_store.as_retriever(
            search_type=settings.SEARCH_TYPE,
            search_kwargs={"k": search_k or settings.SEARCH_K}
        )
    
    def format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            source_info = doc.metadata.get("source_url", "Unknown source")
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted.append(f"[Source: {source_info}]\n{content}")
        return "\n\n---\n\n".join(formatted)
    
    def create_qa_chain(self, retriever, strict: bool = True):
        if strict:
            template = """You are a precise assistant. 
Use ONLY the retrieved context to answer the question. 
If the context doesn't contain the answer, clearly state "I cannot answer this question based on the provided information."

Context: {context}

Question: {question}

Answer:"""
        else:
            template = """You are a helpful assistant. 
Use the retrieved context to answer the question. 
You may supplement with general knowledge but indicate what comes from context vs general knowledge.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        qa_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain
    
    def answer_question(self, vector_store: Chroma, question: str, 
                        strict: bool = True, search_k: int = None) -> Dict:
        try:
            if not vector_store:
                return {
                    "success": False,
                    "answer": "No vector store available. Please ingest data first.",
                    "sources": [],
                    "error": "No vector store"
                }
            
            retriever = self.create_retriever(vector_store, search_k)
            qa_chain = self.create_qa_chain(retriever, strict)
            
            relevant_docs = retriever.invoke(question)
            answer = qa_chain.invoke(question)
            sources = list(set(doc.metadata.get("source_url", "Unknown") for doc in relevant_docs))
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "relevant_docs_count": len(relevant_docs),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "answer": "",
                "sources": [],
                "relevant_docs_count": 0,
                "error": str(e)
            }
