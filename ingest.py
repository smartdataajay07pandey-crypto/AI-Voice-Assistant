import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load API Key
load_dotenv()

def run_ingestion():
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("Please put your resource files in the 'data/' folder and run again.")
        return

    # 1. Load Documents (PDFs and Text files)
    print("Loading resources from ./data...")
    pdf_loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader('./data', glob="./*.txt", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + txt_loader.load()

    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "unknown")


    if not docs:
        print(" No documents found in /data.")
        return

    # 2. Chunking (Breaking files into small pieces for the AI to read)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # 3. Vectorization & Storage
    print(f"Embedding {len(chunks)} text chunks into ChromaDB...")
    
    # This creates a local folder called 'database/chroma_db'
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="./database/chroma_db"
    )
    
    print("Knowledge base updated. Your assistant is ready to learn.")

if __name__ == "__main__":
    run_ingestion()