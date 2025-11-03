
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Mock LangChain and ChromaDB components for demonstration
# In a real application, you would install and import these libraries
# pip install langchain chromadb sentence-transformers

# Mocking LangChain components
from langchain_core.documents import Document

from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

# Mocking ChromaDB
from langchain_community.vectorstores import Chroma


# --- FastAPI Application ---

app = FastAPI()

# --- RAG Setup (Global Initialization) ---
VECTOR_STORE = None
EMBEDDING_MODEL = None
COLLECTION_NAME = "rag_documents"

@app.on_event("startup")
def startup_event():
    global VECTOR_STORE, EMBEDDING_MODEL
    load_dotenv() # Load environment variables from .env

    # 1. Initialize Embedding Model
    # Using a local model for HuggingFaceEmbeddings
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Initialize Vector Store (ChromaDB)
    # This will create a persistent ChromaDB in a directory named 'chroma_db'
    VECTOR_STORE = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
    
    # Ensure the collection exists
    # Note: Chroma.from_documents will create the collection if it doesn't exist
    # For an empty startup, we just need to ensure the directory is set up.
    # We can't directly 'get_or_create_collection' without adding documents in the real Chroma.
    # The collection will be created upon the first document upload.

    # 3. Pre-load a mock document for immediate testing (optional, can be removed for production)
    # This part is for initial testing and can be removed once document upload is confirmed.
    if not VECTOR_STORE.get().get('ids'): # Check if the collection is empty
        mock_content = "The Q3 financial results showed a 12% growth in revenue, primarily driven by the new product line. The CEO, Jane Doe, mentioned that the company is on track to exceed its annual targets. The full report will be released next week."
        mock_doc = Document(page_content=mock_content, metadata={"source": "Q3 Financial Report (Page 12)"})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.split_documents([mock_doc])
        
        # Add to ChromaDB. This will create the collection if it doesn't exist.
        Chroma.from_documents(
            documents=chunks, 
            embedding=EMBEDDING_MODEL, 
            collection_name=COLLECTION_NAME, 
            persist_directory="./chroma_db"
        )
        print("Mock document pre-loaded into ChromaDB.")


# --- API Models ---

class UploadResponse(BaseModel):
    status: str
    message: str
    chunks_processed: int

class StatusResponse(BaseModel):
    status: str
    indexed_count: int
    collection_name: str

class ChatQuery(BaseModel):
    query: str

class Source(BaseModel):
    title: str
    uri: str

class ChatResponse(BaseModel):
    response_text: str
    sources: List[Source]


# --- API Endpoints ---

@app.post("/api/v1/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Determine the loader based on file type
        if file.filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # 3. Add to ChromaDB
        # Re-initialize Chroma with persist_directory to ensure documents are added to the persistent store
        db = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
        db.add_documents(chunks)

        return UploadResponse(
            status="success",
            message="Document indexed successfully.",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.get("/api/v1/status", response_model=StatusResponse)
def get_status():
    # Re-initialize Chroma with persist_directory to get the current state
    db = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
    
    # Check if the collection exists and get the count
    try:
        indexed_count = db.get(include=["documents"])["ids"] # Get only IDs to count documents
        indexed_count = len(indexed_count)
    except Exception: # Collection might not exist yet
        indexed_count = 0
    
    return StatusResponse(
        status="ready",
        indexed_count=indexed_count,
        collection_name=COLLECTION_NAME
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_rag(query: ChatQuery):
    # Re-initialize Chroma with persist_directory to ensure we are querying the persistent store
    db = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Mock LLM setup (replace with actual LLM if API key is available)
    # For demonstration, we'll use a simple mock or a placeholder.
    # If you have an OpenAI API key, you can use: llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0) # Placeholder, requires GOOGLE_API_KEY in .env

    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
    )

    try:
        response = rag_chain.invoke(query.query)
        response_text = response.content

        # Extract sources from the retriever's last run (if possible, or mock)
        # LangChain's retriever doesn't directly expose the last retrieved docs in this chain setup.
        # For a real implementation, you'd need to modify the chain to return sources.
        # For now, we'll mock source extraction or rely on metadata if available in the response.
        
        # A more robust way to get sources would be to run retriever.get_relevant_documents(query.query)
        # separately and then pass both context and sources to the LLM chain.
        retrieved_docs = retriever.get_relevant_documents(query.query)
        sources = []
        for doc in retrieved_docs:
            source_title = doc.metadata.get("source", doc.metadata.get("file_name", "Unknown Document"))
            # You might want to parse page numbers from metadata if available from PDF loader
            if "page" in doc.metadata:
                source_title += f" (Page {doc.metadata["page"] + 1})"
            sources.append(Source(title=source_title, uri="#"))

        return ChatResponse(
            response_text=response_text,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG chain execution: {str(e)}")

# To run this application:
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi "uvicorn[standard]"
# 2. Save the code as rag_backend.py
# 3. Run the server:
#    uvicorn rag_backend:app --reload
