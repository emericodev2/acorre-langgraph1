import re
from urllib.parse import urlparse

def generate_collection_name(url: str) -> str:
    domain = urlparse(url).netloc.replace('.', '_')
    return f"web_rag_{domain}"

def validate_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|https)://'  
        r'(?:\S+(?::\S*)?@)?'  
        r'(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'  
        r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}'  
        r'(?:\.(?:[0-9]{1,3}))|'  
        r'(?:(?:[a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'  
        r'(?::\d{2,5})?'  
        r'(?:/\S*)?$'
    )
    return re.match(regex, url) is not None

def clean_html_content(content: str) -> str:
    # Remove excessive whitespace and HTML tags if present
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', content)).strip()

def get_page_title(url: str) -> str:
    # Simple title from URL
    return urlparse(url).path.strip("/").replace("_", " ").title()
