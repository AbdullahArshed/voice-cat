import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from pipecat.agent import Agent
from pipecat.input.websocket import WebSocketAudioInput
from pipecat.output.audio import TTSOutput
from pipecat.llm import OpenAIChatLLM
from pipecat.tools import Tool
from pipecat.memory import InMemoryChatMessageHistory

load_dotenv()

app = FastAPI(title="Streaming Voice Agent with Pipecat")

# Configure Pipecat components
openai_llm = OpenAIChatLLM(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

output = TTSOutput()  # Uses system TTS (macOS) or fallback
memory = InMemoryChatMessageHistory()

# Custom tool (optional)
class HelloTool(Tool):
    def description(self) -> str:
        return "Say hello in a friendly way."

    def invoke(self, input_text: str) -> str:
        return "Hello! How can I help you today?"

tools = [HelloTool()]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Initialize WebSocket audio input
    ws_input = WebSocketAudioInput(websocket)

    agent = Agent(
        input=ws_input,
        output=output,
        llm=openai_llm,
        memory=memory,
        tools=tools
    )

    try:
        await agent.run()
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

# Optional: basic UI for testing
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
    <body>
        <h2>🎙️ Pipecat Voice Agent WebSocket</h2>
        <p>Connect via WebSocket client to <code>ws://localhost:8000/ws</code></p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_agent:app", host="0.0.0.0", port=8000, reload=True)
