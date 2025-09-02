# RAG Website Assistant

A powerful Retrieval-Augmented Generation (RAG) system that can scrape website content, store it in a vector database, and provide intelligent responses with OpenAI LLM fallback.

## Features

- 🌐 **Website Scraping**: Automatically scrape and store website content
- 📄 **Document RAG**: Upload PDFs, text files, HTML, XML, Markdown, **CSV, and Excel (XLSX)** files for retrieval-augmented generation.
- 🔍 **Vector Search**: Fast semantic search through stored content
- 🤖 **AI Responses**: Generate contextual answers using RAG content
- 🚀 **LLM Fallback**: Seamlessly fall back to OpenAI when RAG content is insufficient
- 💻 **Web Interface**: Beautiful, responsive web UI for easy interaction
- 📊 **ChromaDB Integration**: Persistent vector storage for your content
- 🗑️ **Delete Content Source**: Remove specific website or document data from ChromaDB.

## How It Works

1.  **Content Ingestion**: Add website URLs or upload documents to scrape and store their content.
2.  **Vector Storage**: Content is processed, chunked, and stored in ChromaDB with embeddings.
3.  **Query Processing**: When you ask a question, the system searches for relevant content across both websites and documents.
4.  **RAG Response**: If relevant content is found, it generates a response using that context.
5.  **LLM Fallback**: If no relevant content exists, it falls back to OpenAI for general knowledge.

## Setup

### Prerequisites

-   Python 3.9+
-   OpenAI API key
-   **`libmagic` (for document type detection)**:
    *   **Windows**: Install `python-magic-bin` by running `pip install python-magic-bin`. It bundles the `libmagic` DLLs. If that fails, you might need to install `msvc-runtime` via `pip install msvc-runtime` first.
    *   **macOS**: `brew install libmagic`
    *   **Linux**: `sudo apt-get update && sudo apt-get install libmagic-dev` (Debian/Ubuntu) or `sudo yum install file-devel` (CentOS/RHEL).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd acorre-langgraph
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=your_actual_openai_api_key_here
    ```

### Running the Application

1.  **Start the web server**:
    ```bash
    python app.py
    ```

2.  **Open your browser** and navigate to:
    ```
    http://localhost:8000
    ```

## Usage

### Adding Website Content

1.  In the web interface, enter a website URL in the "Add Website Content" section.
2.  Click "Add Website" to scrape and store the content.

### Uploading Document Content

1.  In the web interface, use the "Upload Document" section.
2.  Click "Choose Document" to select a file (PDF, TXT, HTML, XML, Markdown are best supported).
3.  Click "Upload Document" to process and add its content to the RAG database.

### Querying Content

1.  In the "Conversation with RAG Assistant" section, enter your question.
2.  The system will automatically search for relevant content across both websites and uploaded documents.
3.  If no relevant content is found, it will fall back to OpenAI LLM.

### API Endpoints

The system also provides REST API endpoints:

-   `POST /add-website` - Add a website to the RAG database
-   `POST /upload-document` - Upload a document to the RAG database
-   `POST /query` - Query the RAG system
-   `GET /get-content-sources` - Get list of available content sources

## Architecture

### Core Components

-   **`graph.py`**: LangGraph implementation with RAG workflow (website and document processing)
-   **`app.py`**: FastAPI web application with user interface
-   **ChromaDB**: Vector database for storing embeddings
-   **OpenAI**: LLM provider for text generation and embeddings
-   **`pypdf` / `unstructured` / `python-magic`**: Document loading and parsing

### Workflow

```mermaid
graph TD;
    A[User Query] --> B{Content Available?};
    B -- Yes --> C[Search RAG Database (Web + Docs)];
    C --> D[Generate RAG Response];
    B -- No --> E[Fallback to OpenAI LLM];
    D --> F[Send AI Response];
    E --> F;

    G[Add Website URL] --> H[Scrape Website];
    H --> I[Store in Vector DB];

    J[Upload Document] --> K[Process Document];
    K --> I;
```

## Configuration

### Environment Variables

-   `OPENAI_API_KEY`: Your OpenAI API key (required)
-   `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
-   `OPENAI_TEMPERATURE`: Response creativity (default: 0.1)

### Customization

You can modify the following in `graph.py`:
-   Chunk size and overlap for text splitting
-   Number of documents to retrieve (k value)
-   Prompt templates for RAG and fallback responses

## Troubleshooting

### Common Issues

1.  **`libmagic` Error**: Ensure `libmagic` is installed correctly for your OS (see Prerequisites).
2.  **OpenAI API Key Error**: Ensure your `.env` file contains the correct API key.
3.  **Website Scraping Fails**: Some websites may block automated scraping.
4.  **Document Processing Fails**: Ensure the document is readable and not corrupted. `unstructured` might require additional system dependencies for certain file types (e.g., `poppler-utils` for advanced PDF parsing on Linux).
5.  **Vector Database Issues**: Check if ChromaDB has proper permissions.

### Performance Tips

-   Use smaller chunk sizes for more precise retrieval.
-   Adjust the k value based on your content volume.
-   Consider using a more powerful OpenAI model for better responses.

## Development

### Project Structure

```
acorre-langgraph/
├── src/agent/
│   └── graph.py          # RAG system implementation (Web + Docs)
├── app.py                # Web application (Frontend + API)
├── requirements.txt      # Python dependencies
├── static/              # Static assets
├── uploaded_documents/  # Directory for temporary document uploads
└── tests/               # Test files
```

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1.  Extend the `State` class in `graph.py` for new data.
2.  Add new nodes to the LangGraph workflow.
3.  Update the web interface in `app.py`.
4.  Add corresponding API endpoints.

## Deployment on LangGraph Platform (API Usage)

This project is designed to be easily deployed on the LangGraph platform, exposing your RAG system as an API.

### 1. Install Dependencies
Ensure you have the following installed:

*   **Python 3.11+**
*   **Poetry** (if managing dependencies with `pyproject.toml`)
*   **`libmagic` system library**: 
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get install libmagic1`
    *   **macOS**: `brew install libmagic`
    *   **Windows**: Download binaries from [GnuWin32](http://gnuwin32.sourceforge.net/packages/file.htm) (file-X.X.X-bin.zip and file-X.X.X-dep.zip), extract, and add the `bin` directory containing `magic1.dll` to your system's PATH environment variable. Restart your terminal/IDE after adding to PATH.

### 2. Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/acorre-langgraph.git
    cd acorre-langgraph
    ```
2.  **Create a virtual environment and install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
    pip install -r requirements.txt
    ```
    *(Note: If you are using `pyproject.toml` with Poetry, you might use `poetry install` instead of `pip install -r requirements.txt`.)*

3.  **Set up OpenAI API Key**:
    Create a `.env` file in the project root and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

### 3. Running the Application Locally

To start the FastAPI application with the RAG Chat Dashboard locally:

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:8000` to interact with the dashboard.

### 4. Deployment on LangGraph Platform (and other Docker environments)

To deploy this application, especially if `libmagic` is a requirement in the deployment environment (e.g., LangGraph platform, Docker containers), you will need to build a Docker image. A `Dockerfile` has been provided in the project root to facilitate this.

**A. Build the Docker Image**

Navigate to the root of your project directory in your terminal and run:

```bash
docker build -t acorre-langgraph-rag-app .
```

**B. Run the Docker Container (for local testing of Docker image)**

To test your Docker image locally, you can run it:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY="your_openai_api_key_here" acorre-langgraph-rag-app
```

Then access `http://localhost:8000` in your browser.

**C. Deploy to LangGraph Platform**

For deployment to the LangGraph platform, follow their specific documentation for deploying custom graphs or applications using a `Dockerfile`. Typically, this involves:

1.  **Connecting your repository** to the LangGraph platform.
2.  Ensuring the `Dockerfile` is at the root of your repository.
3.  Configuring **environment variables** (like `OPENAI_API_KEY`) within the LangGraph platform's UI, rather than embedding them directly in the `Dockerfile` or `CMD` for security.
4.  Initiating the **deployment process**, which will use your `Dockerfile` to build and run your application.

**Important Note**: The `CMD ["python", "app.py"]` in the `Dockerfile` assumes `app.py` is the entry point for your web UI. If the LangGraph platform primarily expects to run *only* the LangGraph agent for API purposes and has its own UI/server, you might need to adjust the `CMD` or how you integrate. However, for a self-contained web app, this `Dockerfile` setup for `app.py` should work.

### 5. API Endpoints

Your application exposes the following API endpoints:
