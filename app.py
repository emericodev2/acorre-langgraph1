import os
import json
from langchain.schema import HumanMessage, AIMessage
from agent.graph import graph  # 👈 import graph from graph.py

HISTORY_FILE = "chat_history.json"

# ---------------------------
# Save & Load Chat History
# ---------------------------
def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
                else {"type": "ai", "content": m.content}
                for m in messages
            ],
            f,
            ensure_ascii=False,
            indent=2
        )

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = []
    for item in data:
        if item["type"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages

# ---------------------------
# State Initialization
# ---------------------------
state = {
    "messages": load_history(),
    "website_url": "https://emerico.com",   # 👈 replace with your website
    "retrieved_docs": []
}

# Scrape website only on first run
if not os.path.exists("./chroma_db"):
    print("🔎 Scraping website and creating vectorstore...")
    state = graph.invoke(state)
    print("✅ Website scraped and indexed.")
else:
    print("📂 Using existing vectorstore (chroma_db).")

# ---------------------------
# Chat Loop
# ---------------------------
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        save_history(state["messages"])
        print("💾 Chat history saved. Goodbye!")
        break

    state["messages"].append(HumanMessage(content=user_input))
    state = graph.invoke(state)

    bot_reply = state["messages"][-1].content
    print(f"Bot: {bot_reply}")

    save_history(state["messages"])
