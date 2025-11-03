
import os
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- RAG Setup (Global Initialization) ---
VECTOR_STORE = None
EMBEDDING_MODEL = None
COLLECTION_NAME = "rag_documents"

@app.on_event("startup")
def startup_event():
    global VECTOR_STORE, EMBEDDING_MODEL
    logger.info("Executing startup event.")
    load_dotenv() # Load environment variables from .env

    # 1. Initialize Embedding Model
    logger.info("Initializing embedding model.")
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Initialize Vector Store (ChromaDB)
    logger.info("Initializing vector store.")
    VECTOR_STORE = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL, collection_name=COLLECTION_NAME)
    
    # 3. Pre-load a mock document for immediate testing (optional, can be removed for production)
    if not VECTOR_STORE.get().get('ids'): # Check if the collection is empty
        logger.info("No existing documents found in vector store. No mock document to pre-load.")
    else:
        logger.info("Existing documents found in vector store. Skipping mock document loading.")


# --- API Models ---

class UploadResponse(BaseModel):
    status: str
    message: str
    files_processed: int

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
    logger.info(f"Received request to upload document: {file.filename}")
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {file_path}")

        # Determine the loader based on file type
        if file.filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = file.filename
        logger.info(f"Loaded {len(documents)} documents from file.")

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks.")

        # 3. Add to ChromaDB
        VECTOR_STORE.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to ChromaDB.")

        return UploadResponse(
            status="success",
            message="Document indexed successfully.",
            files_processed=1
        )
    except Exception as e:
        logger.error(f"Error processing document upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


@app.get("/api/v1/status", response_model=StatusResponse)
def get_status():
    logger.info("Received request for indexing status.")
    try:
        metadatas = VECTOR_STORE.get(include=["metadatas"])["metadatas"]
        unique_sources = set(m["source"] for m in metadatas)
        indexed_count = len(unique_sources)
        logger.info(f"Current indexed document count: {indexed_count}")
        status = "ready"
    except Exception as e:
        logger.error(f"Error getting indexing status: {e}", exc_info=True)
        indexed_count = 0
        status = "error"
    
    return StatusResponse(
        status=status,
        indexed_count=indexed_count,
        collection_name=COLLECTION_NAME
    )

class DocumentInfo(BaseModel):
    id: str
    name: str
    status: str = "indexed"

@app.get("/api/v1/documents", response_model=List[DocumentInfo])
def get_documents():
    logger.info("Received request to get indexed documents.")
    try:
        # Retrieve all documents and their metadata
        retrieved_data = VECTOR_STORE.get(include=["metadatas"])
        
        # Use a set to store unique source names to avoid duplicates
        unique_sources = set()
        
        # The 'metadatas' key should contain a list of metadata dictionaries
        if "metadatas" in retrieved_data and retrieved_data["metadatas"]:
            for metadata in retrieved_data["metadatas"]:
                source = metadata.get("source")
                if source:
                    # Remove the temporary directory prefix if it exists
                    if "temp_docs" in source:
                        source = os.path.basename(source)
                    unique_sources.add(source)
        
        # Convert the set of unique sources to the desired response format
        documents = [
            DocumentInfo(id=str(i), name=source)
            for i, source in enumerate(unique_sources)
        ]
        logger.info(f"Found {len(documents)} indexed documents.")
        return documents
    except Exception as e:
        logger.error(f"Error getting documents: {e}", exc_info=True)
        # If there's an error (e.g., collection doesn't exist), return an empty list
        return []


@app.get("/api/v1/chat")
async def chat_get():
    logger.warning("GET request received for /api/v1/chat, which only supports POST.")
    return {"message": "This endpoint only supports POST requests."}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_rag(query: ChatQuery):
    logger.info(f"Received chat query: {query.query}")
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})

    # Mock LLM setup (replace with actual LLM if API key is available)
    # For demonstration, we'll use a simple mock or a placeholder.
    # If you have an OpenAI API key, you can use: llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Placeholder, requires GOOGLE_API_KEY in .env

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
        logger.info(f"Generated response: {response_text}")

        # A more robust way to get sources would be to run retriever.get_relevant_documents(query.query)
        # separately and then pass both context and sources to the LLM chain.
        retrieved_docs = retriever.get_relevant_documents(query.query)
        sources = []
        for doc in retrieved_docs:
            source_title = doc.metadata.get("source", doc.metadata.get("file_name", "Unknown Document"))
            # Remove the temporary directory prefix if it exists
            if "temp_docs" in source_title:
                source_title = os.path.basename(source_title)
            # You might want to parse page numbers from metadata if available from PDF loader
            if "page" in doc.metadata:
                source_title += f" (Page {doc.metadata['page'] + 1})"
            sources.append(Source(title=source_title, uri="#"))
        logger.info(f"Retrieved {len(sources)} sources.")

        return ChatResponse(
            response_text=response_text,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error during RAG chain execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during RAG chain execution: {str(e)}")

# To run this application:
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi "uvicorn[standard]"
# 2. Save the code as rag_backend.py
# 3. Run the server:
#    uvicorn rag_backend:app --reload
