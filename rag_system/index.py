import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

CONNECTION = os.getenv("DATABASE_URL")

documents = []

for file in os.listdir("docs"):
    if file.endswith(".pdf"):

        loader = PyPDFLoader(f"docs/{file}")

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

# Split into chunks
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# Create embedding model 
embeddings = GoogleGenerativeAIEmbeddings( 
    model="gemini-embedding-001" )

# Store in vector DB
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    connection=CONNECTION,
    collection_name="rag_collection"
)

print("Indexing Complete")