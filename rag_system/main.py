from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

loader = PyPDFLoader("cv.pdf")

documents = loader.load()

# Split into chunks
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:\n")
    print(doc.page_content)
    print("=" * 50)
# Create embedding model 
embeddings = GoogleGenerativeAIEmbeddings( 
    model="gemini-embedding-001" )

# Store in vector DB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

query = "What is occupation of ayush " 

retrieved_docs = retriever.invoke(query)
context = "\n".join([
    doc.page_content for doc in retrieved_docs
])

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

response = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(response.content)

