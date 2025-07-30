import os
import sys
import json
import asyncio
import base64
from io import BytesIO
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from loguru import logger
from gtts import gTTS

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

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required in .env")

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
                # Convert raw audio to gTTS format and encode
                tts = gTTS(text=frame.text if hasattr(frame, 'text') else "", lang='en')
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_b64 = base64.b64encode(audio_buffer.read()).decode()
                
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
    """Transcribe audio chunk using OpenAI Whisper API"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        # Skip very small audio chunks
        if len(audio_data) < 2000:  # Less than 2KB probably not useful for speech
            logger.info(f"Skipping small audio chunk: {len(audio_data)} bytes")
            return ""
        
        # Create a temporary file-like object with appropriate extension
        audio_file = BytesIO(audio_data)
        
        # Detect audio format and set appropriate filename
        if audio_data[:4] == b'RIFF':
            audio_file.name = "audio.wav"
            logger.info("Detected WAV format")
        elif audio_data[:4] == b'OggS':
            audio_file.name = "audio.ogg" 
            logger.info("Detected OGG format")
        elif audio_data[:4] == b'\x1aE\xdf\xa3':
            audio_file.name = "audio.webm"
            logger.info("Detected WebM/Matroska format")
        elif b'ftyp' in audio_data[:20]:
            audio_file.name = "audio.mp4"
            logger.info("Detected MP4 format")
        else:
            # Try different extensions for better compatibility
            # Many browsers send WebM with Opus codec
            audio_file.name = "audio.ogg"  # Try OGG first as it often works better
            logger.info(f"Unknown format (first 10 bytes: {audio_data[:10]}), trying as OGG")
        
        logger.info(f"Transcribing audio chunk of {len(audio_data)} bytes")
        
        # Frontend now sends WAV format, so this should work directly
        try:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"  # Specify English for better accuracy
            )
            
            result = response.strip() if response else ""
            logger.info(f"Transcription successful: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return ""


async def stream_llm_response(text: str, websocket: WebSocket, conversation_history: list):
    """Get LLM response and stream it back in chunks"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": text})
        
        # Create streaming chat completion
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
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
                    audio_b64 = await synthesize_audio(current_sentence.strip())
                    if audio_b64:
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "content": audio_b64
                        })
                    current_sentence = ""
        
        # Handle any remaining text
        if current_sentence.strip():
            audio_b64 = await synthesize_audio(current_sentence.strip())
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
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "user_speech":
                    # Process transcribed text from frontend
                    try:
                        user_text = message["content"].strip()
                        if user_text:
                            logger.info(f"✅ User said: '{user_text}'")
                            
                            # Get AI response
                            await get_ai_response(user_text, websocket, conversation_history)
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


async def get_ai_response(user_text: str, websocket: WebSocket, conversation_history: list):
    """Get AI response and send it back"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"🤖 Getting AI response for: '{user_text}'")
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_text})
        
        # Get AI response
        chat_completion = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant in a voice call. Keep responses conversational and concise (1-2 sentences max)."}
            ] + conversation_history[-10:],  # Keep last 10 messages
            max_tokens=100,  # Shorter responses for better voice experience
            temperature=0.7
        )
        
        response_text = chat_completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": response_text})
        
        logger.info(f"✅ AI response: '{response_text}'")
        
        # Send AI response text
        await websocket.send_json({
            "type": "ai_response",
            "content": response_text
        })

        # Generate and send audio
        logger.info("🔊 Generating audio response...")
        audio_b64 = await synthesize_audio(response_text)
        if audio_b64:
            logger.info("✅ Audio response generated and sent")
            await websocket.send_json({
                "type": "audio_response",
                "content": audio_b64
            })
        else:
            logger.warning("⚠️ Failed to generate audio response")
        
        logger.info("🎯 AI response complete - ready for next user input")
        
    except Exception as e:
        logger.error(f"❌ AI response error: {e}")
        await websocket.send_json({
            "type": "error",
            "content": "Failed to get AI response"
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


async def synthesize_audio(text: str) -> str:
    """Generate audio from text using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_b64 = base64.b64encode(audio_buffer.read()).decode()
        return audio_b64
    except Exception as e:
        logger.error(f"Audio synthesis error: {e}")
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
        return False
    
    with open(env_path) as f:
        env_content = f.read()
        if 'OPENAI_API_KEY' not in env_content:
            print("❌ OPENAI_API_KEY not found in .env file")
            return False
    
    print("✅ Environment file configured")
    return True


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Startup banner and checks
    print("🎙️ Voice Chat with AI")
    print("=" * 40)
    
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
    
    print(f"🚀 Starting voice chat server...")
    print(f"📱 Open in browser: http://{config.host}:{config.port}/")
    print(f"🎤 Talk with AI at: http://localhost:{config.port}/")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)

    uvicorn.run(
        "stream:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    ) 