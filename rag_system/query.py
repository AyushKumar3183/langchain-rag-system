from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)
chat_history = []

while True:

 query = input("Ask Question: ")

 chat_history.append(f"User: {query}")

 retrieved_docs = retriever.invoke(query)

 context = "\n".join([
    doc.page_content for doc in retrieved_docs
])

 prompt = f"""
 Use the conversation history and context below
to answer the user's question.

Conversation History:
{chat_history}

Context:
{context}

Question:
{query}
"""
 llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

 response = llm.invoke(prompt)

 print("\nAnswer:\n")
 print(response.content)

 chat_history.append(f"Assistant: {response.content}")