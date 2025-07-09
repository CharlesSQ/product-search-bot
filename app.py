from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from conversational_agent import ConversationalAgent
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Product Search Bot",
    description="A conversational agent for product search.",
    version="1.0.0"
)

# Data model for user input
class UserInput(BaseModel):
    message: str

# Conversational agent setup
# Make sure the OPENAI_API_KEY environment variable is set
llm_chat = ChatOpenAI(temperature=0.9, model='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set conversation memory buffer
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5)

agent = ConversationalAgent(
    llm_model=llm_chat, memory=memory)

@app.post("/chat")
async def chat(user_input: UserInput):
    """
    Endpoint to receive user inputs and process them with the conversational agent.
    """
    response = agent.run(user_input.message)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the product search bot. Use the /chat endpoint to interact."}

# To run the application, use the following command in your terminal:
# pip install -r requirements.txt
# uvicorn app:app --reload
