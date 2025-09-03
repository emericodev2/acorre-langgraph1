# chatbot.py
from langchain.schema import HumanMessage
from agent.graph import graph   # replace with actual filename

class WebsiteRAGChatbot:
    def __init__(self, website_url: str):
        self.website_url = website_url
        self.state = {
            "messages": [],
            "website_url": self.website_url,
            "retrieved_docs": []
        }

    def ask(self, user_message: str) -> str:
        self.state["messages"].append(HumanMessage(content=user_message))
        result = graph.invoke(self.state)
        self.state = result
        return result["messages"][-1].content
