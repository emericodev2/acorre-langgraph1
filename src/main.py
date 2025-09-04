# src/main.py

# Import both graphs so LangGraph CLI can see them
from .ingest_graph import graph_ingest
from .query_graph import graph_query

__all__ = ["graph_ingest", "graph_query"]
