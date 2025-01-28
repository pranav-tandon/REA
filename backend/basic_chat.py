from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.llms import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

class ChatRequest(BaseModel):
    user_input: str

# Initialize Ollama with correct endpoint
try:
    ollama_llm = Ollama(
        base_url="http://localhost:11434",
        model=os.getenv("OLLAMA_MODEL_NAME", "llama2"),
        temperature=0
    )
except Exception as e:
    print(f"Failed to initialize LLM: {str(e)}")
    ollama_llm = None

@router.post("/basic-chat")
async def basic_chat_route(request: ChatRequest):
    try:
        if not ollama_llm:
            raise HTTPException(
                status_code=503,
                detail="LLM not initialized. Please ensure Ollama is running."
            )
            
        print(f"Received request: {request}")
        
        # Use Ollama directly
        response = ollama_llm(request.user_input)
        return {"response": str(response)}
        
    except Exception as e:
        error_msg = str(e)
        print(f"LLM Error: {type(e).__name__} - {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {error_msg}"
        ) 