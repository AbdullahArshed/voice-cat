import asyncio
import json
import os
import io
import base64
import websockets
import time
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()

# Initialize APIs
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Conversation Agent")

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# Conversation history storage
conversations: Dict[str, list] = {}

async def get_ai_response(user_input: str, client_id: str) -> tuple[str, float]:
    """Get AI response from OpenAI and return response with timing"""
    try:
        openai_start_time = time.time()
        
        # Initialize conversation if not exists
        if client_id not in conversations:
            conversations[client_id] = [
                {"role": "system", "content": "You are a helpful, friendly voice assistant. Keep your responses concise and conversational, typically 1-2 sentences unless more detail is specifically requested."}
            ]
        
        # Add user message
        conversations[client_id].append({"role": "user", "content": user_input})
        
        # Get AI response
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=conversations[client_id],
            max_tokens=150,
            temperature=0.7
        )
        
        openai_end_time = time.time()
        openai_latency = (openai_end_time - openai_start_time) * 1000  # Convert to milliseconds
        
        ai_response = response.choices[0].message.content.strip()
        
        # Add AI response to conversation
        conversations[client_id].append({"role": "assistant", "content": ai_response})
        
        # Keep conversation history manageable (last 10 messages)
        if len(conversations[client_id]) > 11:  # system + 10 messages
            conversations[client_id] = conversations[client_id][:1] + conversations[client_id][-10:]
        
        logger.info(f"OpenAI API latency: {openai_latency:.2f}ms")
        return ai_response, openai_latency
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now.", 0.0

async def text_to_speech_streaming(text: str, websocket_client: WebSocket, client_id: str, openai_latency: float):
    """Convert text to speech using ElevenLabs WebSocket streaming for ultra-low latency"""
    try:
        tts_start_time = time.time()
        
        # Use a fast, high-quality voice optimized for low latency
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        # Use eleven_flash_v2_5 for lowest latency (around 75ms)
        model_id = "eleven_flash_v2_5"
        
        # WebSocket URI for streaming
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&inactivity_timeout=180"
        
        async with websockets.connect(uri) as elevenlabs_ws:
            connection_time = time.time()
            websocket_connection_latency = (connection_time - tts_start_time) * 1000
            logger.info(f"ElevenLabs WebSocket connection latency: {websocket_connection_latency:.2f}ms")
            
            # Send initial configuration with optimized settings for low latency
            await elevenlabs_ws.send(json.dumps({
                "text": " ",  # Initial space to establish connection
                "voice_settings": {
                    "stability": 0.4,  # Lower for faster generation
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": False  # Disable for lower latency
                },
                "generation_config": {
                    # Aggressive chunking for ultra-low latency
                    "chunk_length_schedule": [50, 90, 120, 150]  # Smaller chunks for faster response
                },
                "xi_api_key": ELEVENLABS_API_KEY,
            }))
            
            # Create timing context for audio generation
            timing_context = {
                "tts_start_time": tts_start_time,
                "connection_time": connection_time,
                "websocket_connection_latency": websocket_connection_latency,
                "openai_latency": openai_latency,
                "first_chunk_received": False,
                "request_start_time": tts_start_time  # Default to tts_start_time if not provided
            }
            
            # Create tasks for sending text and receiving audio
            send_task = asyncio.create_task(send_text_to_elevenlabs(elevenlabs_ws, text))
            receive_task = asyncio.create_task(receive_audio_from_elevenlabs(elevenlabs_ws, websocket_client, client_id, timing_context))
            
            # Wait for both tasks to complete
            await asyncio.gather(send_task, receive_task)
            
    except Exception as e:
        logger.error(f"Error in WebSocket text-to-speech streaming: {e}")
        await manager.send_message({
            "type": "error",
            "message": "Failed to generate audio response"
        }, client_id)

async def text_to_speech_streaming_with_request_timing(text: str, websocket_client: WebSocket, client_id: str, openai_latency: float, request_start_time: float):
    """Wrapper function to include request start time in timing context"""
    try:
        tts_start_time = time.time()
        
        # Use a fast, high-quality voice optimized for low latency
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        # Use eleven_flash_v2_5 for lowest latency (around 75ms)
        model_id = "eleven_flash_v2_5"
        
        # WebSocket URI for streaming
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&inactivity_timeout=180"
        
        async with websockets.connect(uri) as elevenlabs_ws:
            connection_time = time.time()
            websocket_connection_latency = (connection_time - tts_start_time) * 1000
            logger.info(f"ElevenLabs WebSocket connection latency: {websocket_connection_latency:.2f}ms")
            
            # Send initial configuration with optimized settings for low latency
            await elevenlabs_ws.send(json.dumps({
                "text": " ",  # Initial space to establish connection
                "voice_settings": {
                    "stability": 0.4,  # Lower for faster generation
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": False  # Disable for lower latency
                },
                "generation_config": {
                    # Aggressive chunking for ultra-low latency
                    "chunk_length_schedule": [50, 90, 120, 150]  # Smaller chunks for faster response
                },
                "xi_api_key": ELEVENLABS_API_KEY,
            }))
            
            # Create timing context for audio generation with request start time
            timing_context = {
                "tts_start_time": tts_start_time,
                "connection_time": connection_time,
                "websocket_connection_latency": websocket_connection_latency,
                "openai_latency": openai_latency,
                "first_chunk_received": False,
                "request_start_time": request_start_time  # Include the original request start time
            }
            
            # Create tasks for sending text and receiving audio
            send_task = asyncio.create_task(send_text_to_elevenlabs(elevenlabs_ws, text))
            receive_task = asyncio.create_task(receive_audio_from_elevenlabs(elevenlabs_ws, websocket_client, client_id, timing_context))
            
            # Wait for both tasks to complete
            await asyncio.gather(send_task, receive_task)
            
    except Exception as e:
        logger.error(f"Error in WebSocket text-to-speech streaming: {e}")
        await manager.send_message({
            "type": "error",
            "message": "Failed to generate audio response"
        }, client_id)

async def send_text_to_elevenlabs(elevenlabs_ws, text: str):
    """Send text to ElevenLabs WebSocket"""
    try:
        # Send the text in chunks for streaming (optional: can send all at once for short text)
        chunk_size = 100
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in text_chunks:
            await elevenlabs_ws.send(json.dumps({
                "text": chunk,
                "flush": len(chunk) < chunk_size  # Flush on last chunk
            }))
            await asyncio.sleep(0.01)  # Small delay between chunks
        
        # Send empty string to indicate end of text and close connection
        await elevenlabs_ws.send(json.dumps({"text": ""}))
        
    except Exception as e:
        logger.error(f"Error sending text to ElevenLabs: {e}")

async def receive_audio_from_elevenlabs(elevenlabs_ws, websocket_client: WebSocket, client_id: str, timing_context: dict):
    """Receive audio chunks from ElevenLabs and stream to client with timing measurements"""
    try:
        audio_chunks = []
        
        while True:
            try:
                message = await elevenlabs_ws.recv()
                data = json.loads(message)
                
                if data.get("audio"):
                    current_time = time.time()
                    
                    # Decode audio chunk
                    audio_chunk = base64.b64decode(data["audio"])
                    audio_chunks.append(audio_chunk)
                    
                    # Calculate time to first chunk if this is the first audio
                    if not timing_context["first_chunk_received"]:
                        timing_context["first_chunk_received"] = True
                        time_to_first_chunk = (current_time - timing_context["tts_start_time"]) * 1000
                        total_round_trip = (current_time - timing_context.get("request_start_time", timing_context["tts_start_time"])) * 1000
                        
                        logger.info(f"Time to first audio chunk: {time_to_first_chunk:.2f}ms")
                        logger.info(f"Total round-trip time: {total_round_trip:.2f}ms")
                        
                        # Send latency measurements to client
                        await manager.send_message({
                            "type": "latency_measurement",
                            "openai_latency": timing_context["openai_latency"],
                            "websocket_connection_latency": timing_context["websocket_connection_latency"],
                            "time_to_first_chunk": time_to_first_chunk,
                            "total_round_trip": total_round_trip
                        }, client_id)
                    
                    # Stream individual chunks to client for real-time playback
                    await manager.send_message({
                        "type": "audio_chunk",
                        "audio": data["audio"],  # Send base64 encoded chunk
                        "is_final": False
                    }, client_id)
                    
                elif data.get('isFinal'):
                    # Send final message with complete audio
                    complete_audio = b''.join(audio_chunks)
                    audio_base64 = base64.b64encode(complete_audio).decode('utf-8')
                    
                    total_generation_time = (time.time() - timing_context["tts_start_time"]) * 1000
                    logger.info(f"Total audio generation time: {total_generation_time:.2f}ms")
                    
                    await manager.send_message({
                        "type": "audio_response",
                        "audio": audio_base64,
                        "is_final": True,
                        "total_generation_time": total_generation_time
                    }, client_id)
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("ElevenLabs WebSocket connection closed")
                break
                
    except Exception as e:
        logger.error(f"Error receiving audio from ElevenLabs: {e}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "user_speech":
                request_start_time = time.time()
                user_text = message["text"]
                logger.info(f"Received from {client_id}: {user_text}")
                
                # Send acknowledgment
                await manager.send_message({
                    "type": "processing",
                    "message": "Processing your request...",
                    "request_start_time": request_start_time * 1000  # Convert to milliseconds for frontend
                }, client_id)
                
                # Get AI response with timing
                ai_response, openai_latency = await get_ai_response(user_text, client_id)
                
                # Send text response
                await manager.send_message({
                    "type": "ai_response",
                    "text": ai_response
                }, client_id)
                
                # Generate and stream audio using WebSocket streaming for ultra-low latency
                # Update timing context to include request start time
                await text_to_speech_streaming_with_request_timing(ai_response, websocket, client_id, openai_latency, request_start_time)
            
            elif message["type"] == "ping":
                await manager.send_message({
                    "type": "pong"
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Clean up conversation history
        if client_id in conversations:
            del conversations[client_id]
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

@app.get("/")
async def get_voice_agent():
    """Serve the voice agent HTML page"""
    try:
        with open("11.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Voice agent HTML file not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "elevenlabs_configured": bool(os.getenv("ELEVENLABS_API_KEY")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.error("ELEVENLABS_API_KEY not found in environment variables")
        exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    logger.info("Starting Voice Conversation Agent...")
    uvicorn.run(
        "11:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 