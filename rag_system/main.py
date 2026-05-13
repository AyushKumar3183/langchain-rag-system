from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from jose import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_postgres import PGVector
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

security = HTTPBearer()





CONNECTION = os.getenv("DATABASE_URL")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

fake_users = {
    "ayush": {
        "username": "ayush",
        "password": "1234"
    }
}

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vectorstore = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name="rag_collection"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


app = FastAPI()

class QueryRequest(BaseModel):

    question: str

class LoginRequest(BaseModel):

    username: str
    password: str


def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(hours=1)

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )    

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload

    except:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

@app.post("/login")
async def login(request: LoginRequest):

    user = fake_users.get(
        request.username
    )

    if not user:

        return {
            "error": "Invalid username"
        }

    if user["password"] != request.password:

        return {
            "error": "Invalid password"
        }

    token = create_access_token({
        "sub": request.username
    })

    return {
        "access_token": token
    }


@app.post("/ask")
async def ask_question(request: QueryRequest , token: str = Depends(verify_token)):

    docs = retriever.invoke(
        request.question
    )

    context = "\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = f"""
    Answer using the context below.

    Context:
    {context}

    Question:
    {request.question}
    """

    response = await llm.ainvoke(prompt)

    return {
        "answer": response.content
    }