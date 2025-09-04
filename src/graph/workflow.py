from typing import Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from agents.ingestion_agent import DataIngestionAgent
from agents.qa_agent import QAAgent
from langchain_community.vectorstores import Chroma

class AgentState(TypedDict):
    url: str
    question: str
    collection_name: Optional[str]
    documents: List
    vector_store: Optional[Chroma]
    answer: str
    sources: List[str]
    error: str
    success: bool

class WebsiteRAGWorkflow:
    def __init__(self):
        self.ingestion_agent = DataIngestionAgent()
        self.qa_agent = QAAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("validate_input", self.validate_input_node)
        workflow.add_node("ingest_data", self.ingest_data_node)
        workflow.add_node("answer_question", self.answer_question_node)
        workflow.add_node("handle_error", self.handle_error_node)
        workflow.add_node("final_response", self.final_response_node)
        
        workflow.set_entry_point("validate_input")
        
        workflow.add_conditional_edges(
            "validate_input",
            self.check_validation,
            {"valid": "ingest_data", "invalid": "handle_error"}
        )
        workflow.add_conditional_edges(
            "ingest_data",
            self.check_ingestion_success,
            {"success": "answer_question", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "answer_question",
            self.check_qa_success,
            {"success": "final_response", "error": "handle_error"}
        )
        workflow.add_edge("handle_error", "final_response")
        workflow.add_edge("final_response", END)
        return workflow.compile()
    
    def validate_input_node(self, state: AgentState) -> Dict:
        from ..utils.helpers import validate_url
        errors = []
        if not state.get("url"):
            errors.append("URL is required")
        elif not validate_url(state["url"]):
            errors.append("Invalid URL format")
        if not state.get("question"):
            errors.append("Question is required")
        return {"error": "; ".join(errors) if errors else "", "success": len(errors) == 0}
    
    def ingest_data_node(self, state: AgentState) -> Dict:
        result = self.ingestion_agent.ingest_website(state["url"], state.get("collection_name"))
        return {
            "vector_store": result["vector_store"],
            "collection_name": result["collection_name"],
            "documents": [],
            "error": result["error"],
            "success": result["success"]
        }
    
    def answer_question_node(self, state: AgentState) -> Dict:
        result = self.qa_agent.answer_question(state["vector_store"], state["question"])
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "error": result["error"],
            "success": result["success"]
        }
    
    def handle_error_node(self, state: AgentState) -> Dict:
        return {"error": state.get("error", "Unknown error"), "success": False}
    
    def final_response_node(self, state: AgentState) -> Dict:
        return state
    
    def check_validation(self, state: AgentState) -> str:
        return "valid" if state.get("success", False) else "invalid"
    
    def check_ingestion_success(self, state: AgentState) -> str:
        return "success" if state.get("success", False) else "error"
    
    def check_qa_success(self, state: AgentState) -> str:
        return "success" if state.get("success", False) else "error"
    
    def run_workflow(self, url: str, question: str, collection_name: Optional[str] = None) -> Dict:
        initial_state = AgentState(
            url=url,
            question=question,
            collection_name=collection_name,
            documents=[],
            vector_store=None,
            answer="",
            sources=[],
            error="",
            success=False
        )
        return self.graph.invoke(initial_state)
