
import os
import sys
import json
import base64
import argparse
import asyncio
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from dotenv import load_dotenv
from loguru import logger
from gtts import gTTS
import openai

# Load environment
load_dotenv(override=True)

# Setup logging
logger.add(sys.stderr, level="DEBUG")

# FastAPI instance
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Client Init
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required in .env")

openai_client = openai.AsyncOpenAI(api_key=api_key)

# WebSocket connections
connected_clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    yield
    logger.info("Server shutting down...")


@app.get("/", response_class=HTMLResponse)
async def get_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)


async def synthesize_audio(text: str) -> str:
    """Convert text to base64-encoded audio (mp3)."""
    try:
        tts = gTTS(text=text, lang="en")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return ""


async def get_llm_response(messages: list) -> str:
    """Get response from OpenAI LLM."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("WebSocket connection established")
    
    # Initialize conversation context for this session
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Respond to user messages in a friendly, conversational way. Keep responses concise but informative. Provide clear, actionable answers when possible.",
        }
    ]

    try:
        # Send initial greeting
        await websocket.send_json({"type": "info", "content": "Connected! Click Start to begin voice conversation."})
        
        while True:
            try:
                # Increased timeout for voice interactions
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                logger.info("Timeout reached — sending keepalive")
                await websocket.send_json({"type": "keepalive"})
                continue

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON received"})
                continue

            if message["type"] == "start_recording":
                logger.info("Voice recording started")
                await websocket.send_json({"type": "info", "content": "Listening... Speak now!"})

            elif message["type"] == "voice_chunk":
                # Show voice chunks on screen as they come in
                chunk_text = message.get("content", "")
                await websocket.send_json({"type": "voice_chunk", "content": chunk_text})

            elif message["type"] == "stop_recording":
                # Process the complete transcription
                transcribed_text = message.get("content", "").strip()
                logger.info(f"Processing transcribed text: {transcribed_text}")
                
                if not transcribed_text:
                    await websocket.send_json({"type": "error", "content": "No speech detected. Please try again."})
                    continue
                
                await websocket.send_json({"type": "info", "content": "Processing your message..."})
                
                # Add user message to conversation history
                conversation_history.append({"role": "user", "content": transcribed_text})

                # Process with OpenAI LLM
                try:
                    response_text = await get_llm_response(conversation_history)
                    
                    # Add assistant response to conversation history
                    conversation_history.append({"role": "assistant", "content": response_text})

                    # Send text response
                    await websocket.send_json({"type": "message", "content": response_text})

                    # Generate and send audio
                    audio_b64 = await synthesize_audio(response_text)
                    if audio_b64:
                        await websocket.send_json({"type": "audio", "content": audio_b64})
                    else:
                        await websocket.send_json({"type": "error", "content": "Audio synthesis failed"})
                
                except Exception as e:
                    logger.error(f"LLM processing error: {e}")
                    await websocket.send_json({"type": "error", "content": "Failed to process your message. Please try again."})

            elif message["type"] == "text":
                # Handle manual text input
                logger.debug(f"User text message: {message['content']}")
                conversation_history.append({"role": "user", "content": message["content"]})

                try:
                    response_text = await get_llm_response(conversation_history)
                    conversation_history.append({"role": "assistant", "content": response_text})

                    await websocket.send_json({"type": "message", "content": response_text})

                    audio_b64 = await synthesize_audio(response_text)
                    if audio_b64:
                        await websocket.send_json({"type": "audio", "content": audio_b64})
                
                except Exception as e:
                    logger.error(f"LLM processing error: {e}")
                    await websocket.send_json({"type": "error", "content": "Failed to process your message."})

            elif message["type"] == "ping":
                # Handle keepalive ping
                await websocket.send_json({"type": "pong"})

            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
                await websocket.send_json({"type": "error", "content": f"Unknown message type: {message.get('type')}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": "Server error occurred"})
        except:
            pass
    finally:
        connected_clients.discard(websocket)
        logger.info("WebSocket connection closed")


@app.get("/status")
async def get_status():
    return JSONResponse({
        "status": "running",
        "connected_clients": len(connected_clients)
    })


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FAST_API_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)