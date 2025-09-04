import os
import json

class VectorStoreManager:
    STORAGE_FILE = "./vector_store_metadata.json"
    
    def __init__(self):
        if not os.path.exists(self.STORAGE_FILE):
            with open(self.STORAGE_FILE, "w") as f:
                json.dump({}, f)
    
    def store_collection_info(self, collection_name: str, url: str, document_count: int, metadata: dict):
        with open(self.STORAGE_FILE, "r+") as f:
            data = json.load(f)
            data[collection_name] = {
                "url": url,
                "document_count": document_count,
                "metadata": metadata
            }
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    
    def get_collection_info(self, collection_name: str):
        with open(self.STORAGE_FILE, "r") as f:
            data = json.load(f)
        return data.get(collection_name)
