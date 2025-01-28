from fastapi import APIRouter
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

ollama_llm = ChatOpenAI(
    openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11411"),
    openai_api_key=os.getenv("OPENAI_API_KEY", "unused"),
    model_name=os.getenv("OLLAMA_MODEL_NAME", "llama2"),
    temperature=0
)

@router.post("/basic-chat")
def basic_chat_route(user_message: str):
    # Use ollama_llm to generate a response
    response = ollama_llm.predict(user_message)
    return {"response": response} 