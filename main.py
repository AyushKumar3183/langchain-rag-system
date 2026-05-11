from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

# Define structure
class Explanation(BaseModel):
    topic: str = Field(description="Topic name")
    summary: str = Field(description="Short explanation")

# Convert LLM into structured mode
structured_llm = llm.with_structured_output(Explanation)

response = structured_llm.invoke(
    "Explain transformers in simple words"
)


print(response)