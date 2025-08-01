import os
import sys
import json
import asyncio
import base64
import time
from io import BytesIO
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from loguru import logger

# Pipecat imports for streaming (updated for newer versions)
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.frames.frames import (
        Frame, AudioRawFrame, TextFrame, LLMMessagesFrame, 
        TranscriptionFrame, TTSAudioRawFrame, StartInterruptionFrame,
        StopInterruptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    PIPECAT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pipecat not available for streaming: {e}")
    PIPECAT_AVAILABLE = False
    # Define dummy classes to prevent errors
    class FrameProcessor:
        def __init__(self): pass
        async def process_frame(self, frame, direction): pass
        async def push_frame(self, frame, direction): pass
    class Frame: pass
    class Pipeline: pass
    FrameDirection = None

# Load environment
load_dotenv(override=True)

# Setup logging
logger.add(sys.stderr, level="INFO")

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

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is required in .env")
if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY is required in .env")

# Active streaming sessions
active_sessions: Dict[str, str] = {}


class WebSocketTransport(FrameProcessor):
    """Custom transport for WebSocket communication"""
    
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket
        self._audio_buffer = BytesIO()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Handle different frame types for WebSocket communication
        if isinstance(frame, TextFrame):
            # Send text response to client
            await self._websocket.send_json({
                "type": "llm_response", 
                "content": frame.text
            })
            
        elif isinstance(frame, TTSAudioRawFrame):
            # Convert audio to base64 and send to client
            try:
                # Use ElevenLabs for audio instead of gTTS
                audio_b64 = await synthesize_audio_elevenlabs(frame.text if hasattr(frame, 'text') else "")
                if audio_b64:
                    await self._websocket.send_json({
                        "type": "audio_response",
                        "content": audio_b64
                    })
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                
        elif isinstance(frame, TranscriptionFrame):
            # Send transcription to client for real-time display
            await self._websocket.send_json({
                "type": "transcription",
                "content": frame.text
            })
            
        # Always pass frame downstream
        await self.push_frame(frame, direction)


class StreamingVoiceProcessor(FrameProcessor):
    """Processor for handling streaming voice interactions"""
    
    def __init__(self):
        super().__init__()
        self._current_speech = ""
        self._is_speaking = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self._current_speech = ""
            logger.info("User started speaking")
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            logger.info("User stopped speaking")
            
        elif isinstance(frame, TranscriptionFrame):
            if self._is_speaking:
                self._current_speech += f" {frame.text}"
            
        await self.push_frame(frame, direction)


class AudioStreamProcessor(FrameProcessor):
    """Process incoming audio streams"""
    
    def __init__(self):
        super().__init__()
        self._audio_buffer = []
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            # Process incoming audio data
            self._audio_buffer.append(frame.audio)
            
        await self.push_frame(frame, direction)


async def create_streaming_pipeline(websocket: WebSocket):
    """Create a simplified pipeline for voice conversations"""
    
    if not PIPECAT_AVAILABLE:
        logger.warning("Pipecat not available, using fallback mode")
        return None
    
    try:
        # For now, let's skip the complex pipeline and use direct OpenAI calls
        # This avoids the API compatibility issues
        logger.info("Skipping complex pipeline creation due to API changes")
        return None
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main voice chat interface at root"""
    file_path = os.path.join(os.path.dirname(__file__), "stream.html")
    return FileResponse(file_path)

@app.get("/stream", response_class=HTMLResponse)
async def get_stream_page():
    """Serve the streaming voice chat interface (same as root)"""
    file_path = os.path.join(os.path.dirname(__file__), "stream.html")
    return FileResponse(file_path)


@app.websocket("/stream-chat")
async def websocket_stream_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming voice chat"""
    await websocket.accept()
    session_id = f"session_{id(websocket)}"
    
    # Conversation history for this session
    conversation_history = []
    
    try:
        logger.info(f"Starting streaming session: {session_id}")
        
        # Try to create pipeline, but fall back to simple mode if it fails
        pipeline = await create_streaming_pipeline(websocket)
        
        if pipeline is None:
            # Use simple implementation 
            active_sessions[session_id] = "simple_mode"
            await websocket.send_json({
                "type": "system", 
                "content": "📞 Ready for voice call - click the call button to start!"
            })
            await simple_voice_call_handler(websocket, conversation_history)
            return
        
        # If we get here, the advanced pipeline worked (currently disabled)
        logger.info("Advanced pipeline mode would be used here")
        await simple_voice_call_handler(websocket, conversation_history)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
    finally:
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")


class SpeechBuffer:
    """Buffer for accumulating speech transcription chunks"""
    def __init__(self):
        self.buffer = ""
        self.last_activity = asyncio.get_event_loop().time()
        self.sentence_endings = {'.', '!', '?', '。', '！', '？'}
        
    def add_chunk(self, text: str):
        """Add a transcription chunk to the buffer"""
        self.buffer += " " + text.strip()
        self.last_activity = asyncio.get_event_loop().time()
        
    def has_complete_sentence(self) -> bool:
        """Check if buffer contains a complete sentence"""
        if not self.buffer.strip():
            return False
        return any(ending in self.buffer for ending in self.sentence_endings)
    
    def has_enough_words(self, min_words: int = 3) -> bool:
        """Check if buffer has enough words to process"""
        if not self.buffer.strip():
            return False
        words = self.buffer.strip().split()
        return len(words) >= min_words
        
    def is_timeout(self, timeout_seconds: float = 2.5) -> bool:
        """Check if buffer has timed out (no activity for X seconds)"""
        return (asyncio.get_event_loop().time() - self.last_activity) > timeout_seconds
        
    def get_and_clear(self) -> str:
        """Get buffer content and clear it"""
        content = self.buffer.strip()
        self.buffer = ""
        return content


async def transcribe_audio_chunk(audio_data: bytes) -> str:
    """Transcribe audio chunk using OpenAI Whisper API - OPTIMIZED FOR SPEED"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        
        transcription_start_time = time.time()
        
        # Skip very small audio chunks
        if len(audio_data) < 1500:  # Reduced threshold for faster processing
            print(f"⏱️  Skipping small audio chunk: {len(audio_data)} bytes")
            return ""
        
        # Create a temporary file-like object with appropriate extension
        audio_file = BytesIO(audio_data)
        
        # Detect audio format and set appropriate filename
        if audio_data[:4] == b'RIFF':
            audio_file.name = "audio.wav"
            print(f"⏱️  Detected WAV format")
        elif audio_data[:4] == b'OggS':
            audio_file.name = "audio.ogg" 
            print(f"⏱️  Detected OGG format")
        elif audio_data[:4] == b'\x1aE\xdf\xa3':
            audio_file.name = "audio.webm"
            print(f"⏱️  Detected WebM/Matroska format")
        elif b'ftyp' in audio_data[:20]:
            audio_file.name = "audio.mp4"
            print(f"⏱️  Detected MP4 format")
        else:
            audio_file.name = "audio.ogg"
            print(f"⏱️  Unknown format, trying as OGG")
        
        print(f"⏱️  Starting Whisper transcription of {len(audio_data)} bytes")
        
        # OPTIMIZED Whisper API call
        try:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en",  # Specify English for better speed
                # Remove prompt for speed
            )
            
            transcription_end_time = time.time()
            transcription_duration = (transcription_end_time - transcription_start_time) * 1000
            
            result = response.strip() if response else ""
            print(f"⏱️  Whisper transcription completed in {transcription_duration:.2f}ms: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


async def stream_llm_response(text: str, websocket: WebSocket, conversation_history: list):
    """Get LLM response and stream it back in chunks"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": text})
        
        # Create streaming chat completion
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Keep responses concise and conversational since this is a voice chat."}
            ] + conversation_history[-10:],
            max_tokens=150,
            temperature=0.7,
            stream=True
        )
        
        response_text = ""
        current_sentence = ""
        
        # Process streaming response
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content
                current_sentence += content
                
                # Send partial response for real-time display
                await websocket.send_json({
                    "type": "llm_chunk",
                    "content": content
                })
                
                # Check if we have a complete sentence to synthesize
                if any(ending in current_sentence for ending in {'.', '!', '?', '。', '！', '？'}):
                    # Generate audio for this sentence
                    audio_b64 = await synthesize_audio_elevenlabs(current_sentence.strip())
                    if audio_b64:
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "content": audio_b64
                        })
                    current_sentence = ""
        
        # Handle any remaining text
        if current_sentence.strip():
            audio_b64 = await synthesize_audio_elevenlabs(current_sentence.strip())
            if audio_b64:
                await websocket.send_json({
                    "type": "audio_chunk",
                    "content": audio_b64
                })
        
        # Add full response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Send completion signal
        await websocket.send_json({
            "type": "llm_complete",
            "content": response_text
        })
        
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        await websocket.send_json({
            "type": "error",
            "content": "Failed to process your message."
        })


async def simple_voice_call_handler(websocket: WebSocket, conversation_history: list):
    """Simple voice call handler - process text from browser speech recognition"""
    
    logger.info("🎯 Voice call handler started")
    
    try:
        while True:
            try:
                # Track message reception latency
                msg_start_time = time.time()
                data = await websocket.receive_text()
                msg_receive_time = time.time()
                print(f"⏱️  WebSocket message received in {(msg_receive_time - msg_start_time)*1000:.2f}ms")
                
                message = json.loads(data)
                
                if message["type"] == "user_speech":
                    # Process transcribed text from frontend
                    try:
                        process_start_time = time.time()
                        user_text = message["content"].strip()
                        if user_text:
                            logger.info(f"✅ User said: '{user_text}'")
                            print(f"⏱️  Processing user speech: '{user_text[:50]}...' (len: {len(user_text)})")
                            
                            # Get AI response
                            await get_ai_response_fast(user_text, websocket, conversation_history)
                            
                            process_end_time = time.time()
                            total_process_time = (process_end_time - process_start_time) * 1000
                            print(f"⏱️  TOTAL processing time for user input: {total_process_time:.2f}ms")
                        else:
                            logger.info("🔇 Empty speech received")
                        
                    except Exception as e:
                        logger.error(f"❌ Speech processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "content": f"Speech processing failed: {str(e)}"
                        })
                    
            except asyncio.TimeoutError:
                await websocket.ping()
                logger.debug("WebSocket ping sent")
                
    except WebSocketDisconnect:
        logger.info("📞 Call ended - WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ Call handler error: {e}")
    finally:
        logger.info("🎯 Voice call handler ended")


async def get_ai_response_fast(user_text: str, websocket: WebSocket, conversation_history: list):
    """Get AI response optimized for ULTRA LOW LATENCY"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        
        ai_start_time = time.time()
        logger.info(f"🤖 Getting AI response for: '{user_text}'")
        print(f"⏱️  AI request started at {ai_start_time}")
        
        # DON'T add to conversation history for speed
        # conversation_history.append({"role": "user", "content": user_text})
        
        # Send typing indicator
        typing_time = time.time()
        await websocket.send_json({
            "type": "ai_typing",
            "content": "AI is thinking..."
        })
        print(f"⏱️  Typing indicator sent in {(typing_time - ai_start_time)*1000:.2f}ms")
        
        # ULTRA FAST CONFIGURATION - NO HISTORY, MINIMAL TOKENS, FASTEST MODEL
        stream_start_time = time.time()
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Actually faster for simple responses than gpt-4o-mini
            messages=[
                {"role": "system", "content": "Be brief. One short sentence only."},  # Minimal system prompt
                {"role": "user", "content": user_text}
            ],  # NO CONVERSATION HISTORY FOR SPEED
            max_tokens=25,  # VERY short responses for speed
            temperature=0.1,  # Very low for fastest generation
            stream=True,
            # Speed optimizations
            presence_penalty=0,
            frequency_penalty=0,
        )
        stream_create_time = time.time()
        print(f"⏱️  OpenAI stream created in {(stream_create_time - stream_start_time)*1000:.2f}ms")
        
        response_text = ""
        first_chunk_time = None
        
        # Send initial message container
        await websocket.send_json({
            "type": "ai_response_start",
            "content": ""
        })
        
        # Process streaming response
        chunk_count = 0
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                chunk_time = time.time()
                if first_chunk_time is None:
                    first_chunk_time = chunk_time
                    print(f"⏱️  First AI chunk received in {(first_chunk_time - stream_create_time)*1000:.2f}ms")
                
                content = chunk.choices[0].delta.content
                response_text += content
                chunk_count += 1
                
                # Send chunk for real-time display
                await websocket.send_json({
                    "type": "ai_chunk",
                    "content": content,
                    "full_text": response_text
                })
        
        # Generate audio for complete response (don't wait for sentence endings)
        if response_text.strip():
            audio_start_time = time.time()
            print(f"⏱️  Starting ElevenLabs Flash synthesis for complete response: '{response_text.strip()[:30]}...'")
            
            audio_b64 = await synthesize_audio_elevenlabs(response_text.strip())
            audio_end_time = time.time()
            audio_duration = (audio_end_time - audio_start_time) * 1000
            print(f"⏱️  ElevenLabs synthesis completed in {audio_duration:.2f}ms")
            
            if audio_b64:
                await websocket.send_json({
                    "type": "audio_chunk",
                    "content": audio_b64,
                    "text": response_text.strip()
                })
                print(f"⏱️  Audio chunk sent to client")
        
        # DON'T add to conversation history for speed
        # conversation_history.append({"role": "assistant", "content": response_text})
        
        # Send final response
        final_time = time.time()
        await websocket.send_json({
            "type": "ai_response_complete",
            "content": response_text
        })
        
        # Print comprehensive timing summary
        total_ai_time = (final_time - ai_start_time) * 1000
        
        print(f"⏱️  === ULTRA-FAST AI RESPONSE TIMING ===")
        print(f"⏱️  Total AI processing time: {total_ai_time:.2f}ms")
        print(f"⏱️  OpenAI stream creation: {(stream_create_time - stream_start_time)*1000:.2f}ms")
        print(f"⏱️  Time to first chunk: {(first_chunk_time - stream_create_time)*1000:.2f}ms" if first_chunk_time else "⏱️  No chunks received")
        print(f"⏱️  Total chunks processed: {chunk_count}")
        print(f"⏱️  Response length: {len(response_text)} chars")
        print(f"⏱️  === END TIMING SUMMARY ===")
        
        logger.info(f"✅ AI response complete: '{response_text}'")
        logger.info("🎯 Ready for next user input")
        
    except Exception as e:
        logger.error(f"❌ AI response error: {e}")
        await websocket.send_json({
            "type": "error",
            "content": f"Failed to get AI response: {str(e)}"
        })


@app.get("/stream-status")
async def get_stream_status():
    """Get status of streaming sessions"""
    return {
        "status": "running",
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "streaming_voice_chat"}


async def synthesize_audio_elevenlabs(text: str) -> str:
    """Generate audio using ElevenLabs Flash TTS (75ms latency)"""
    try:
        import httpx
        
        if not text.strip():
            return ""
            
        synthesis_start = time.time()
        print(f"⏱️  Starting ElevenLabs Flash TTS for {len(text)} characters")
        
        # ElevenLabs Flash API endpoint
        url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB"  # Default voice
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_flash_v2_5",  # Flash model for 75ms latency
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=30.0)
            
        api_time = time.time()
        print(f"⏱️  ElevenLabs API call completed in {(api_time - synthesis_start)*1000:.2f}ms")
        
        if response.status_code == 200:
            audio_data = response.content
            audio_b64 = base64.b64encode(audio_data).decode()
            
            encode_time = time.time()
            total_time = (encode_time - synthesis_start) * 1000
            print(f"⏱️  Total ElevenLabs Flash synthesis: {total_time:.2f}ms (audio size: {len(audio_b64)} chars)")
            
            return audio_b64
        else:
            logger.error(f"ElevenLabs API error {response.status_code}: {response.text}")
            return ""
            
    except Exception as e:
        logger.error(f"ElevenLabs synthesis error: {e}")
        return ""


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pipecat
        print("✅ Pipecat found")
        
        try:
            import whisper
            print("✅ Whisper found")
        except ImportError:
            print("⚠️  Whisper not found (optional for advanced features)")
        
        try:
            import torch
            print("✅ Torch found")
        except ImportError:
            print("⚠️  Torch not found (optional for advanced features)")
        
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        print("Note: The streaming server will work with basic features even without all dependencies")
        return True  # Allow running even without all deps


def check_env_file():
    """Check if .env file exists with required variables"""
    from pathlib import Path
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found")
        print("Please create a .env file with:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("ELEVENLABS_API_KEY=your_elevenlabs_api_key_here")
        return False
    
    with open(env_path) as f:
        env_content = f.read()
        if 'OPENAI_API_KEY' not in env_content:
            print("❌ OPENAI_API_KEY not found in .env file")
            return False
        if 'ELEVENLABS_API_KEY' not in env_content:
            print("❌ ELEVENLABS_API_KEY not found in .env file")
            print("Sign up at https://elevenlabs.io/ to get an API key")
            return False
    
    print("✅ Environment file configured")
    return True


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Startup banner and checks
    print("🎙️ Voice Chat with AI - OPTIMIZED FOR SPEED")
    print("=" * 50)
    print("🚀 Using ElevenLabs Flash TTS (75ms latency)")
    print("🚀 Using GPT-4o-mini (111 tokens/sec)")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Run voice chat server')
    parser.add_argument("--host", type=str, default="0.0.0.0", help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument("--port", type=int, default=8001, help='Port to bind to (default: 8001)')
    parser.add_argument("--reload", action="store_true", help='Enable auto-reload for development')
    parser.add_argument("--check-only", action="store_true", help='Only check dependencies and exit')
    config = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    if config.check_only:
        print("✅ All checks passed!")
        sys.exit(0)
    
    print(f"🚀 Starting OPTIMIZED voice chat server...")
    print(f"📱 Open in browser: http://{config.host}:{config.port}/")
    print(f"🎤 Talk with AI at: http://localhost:{config.port}/")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    uvicorn.run(
        "streamlabs:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    ) 