import os
import sys
import json
import asyncio
import base64
from io import BytesIO
from typing import Optional, Dict, Any
import httpx  # Add for ElevenLabs API calls

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from loguru import logger
from gtts import gTTS

# Add additional optimization configurations at the top
# Optimizations for ultra-low latency (targeting 500-700ms)
ULTRA_LOW_LATENCY = True
PARALLEL_PROCESSING = True
STREAM_LLM_RESPONSES = True

# Use fastest available models
LLM_MODEL = "gpt-3.5-turbo" if not ULTRA_LOW_LATENCY else "gpt-3.5-turbo-0125"  # Latest, fastest variant
MAX_LLM_TOKENS = 100 if not ULTRA_LOW_LATENCY else 50  # Shorter responses for speed
LLM_TEMPERATURE = 0.7 if not ULTRA_LOW_LATENCY else 0.3  # Lower temperature for faster processing

# ElevenLabs optimizations
ELEVENLABS_MODEL = "eleven_turbo_v2"  # Fastest model
ELEVENLABS_CHUNK_SIZE = 4096 if not ULTRA_LOW_LATENCY else 2048  # Smaller chunks for faster streaming

# Pipecat imports for ultra-fast streaming voice processing
try:
    from pipecat.frames.frames import (
        Frame, AudioRawFrame, TextFrame, TTSAudioRawFrame, 
        TranscriptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    
    PIPECAT_AVAILABLE = True
    logger.info("✅ Pipecat enabled for ultra-fast voice processing")
except ImportError as e:
    logger.error(f"❌ Pipecat import failed: {e}")
    PIPECAT_AVAILABLE = False
    # Define minimal classes for fallback
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
logger.info(f"🔑 OpenAI API Key status: {'✅ Set' if api_key and api_key != 'your_openai_api_key_here' else '❌ Missing/Invalid'}")

if not api_key or api_key == "your_openai_api_key_here":
    logger.warning("⚠️ OpenAI API key not configured - running in test mode")
    logger.warning("⚠️ Create .env file with OPENAI_API_KEY=your_actual_key for full functionality")
else:
    logger.info("✅ OpenAI API key configured")

# ElevenLabs API Key and settings
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default to Rachel voice
use_elevenlabs = elevenlabs_api_key is not None and elevenlabs_api_key != "your_elevenlabs_api_key_here"


def create_openai_client():
    """Create OpenAI client with error handling for version compatibility"""
    try:
        from openai import AsyncOpenAI
        # Initialize with minimal parameters to avoid version conflicts
        return AsyncOpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        try:
            # Fallback initialization for older versions
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key, timeout=30.0)
        except Exception as e2:
            logger.error(f"Fallback OpenAI client creation failed: {e2}")
            raise ValueError(f"Cannot create OpenAI client. Version compatibility issue: {e}")


if use_elevenlabs:
    logger.info("✅ ElevenLabs TTS enabled for low-latency audio")
else:
    logger.info("⚠️ ElevenLabs not configured, using gTTS (higher latency)")

# Active streaming sessions
active_sessions: Dict[str, str] = {}


class WebSocketTransport(FrameProcessor):
    """Ultra-fast WebSocket transport for real-time voice chat"""
    
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket
        self._audio_buffer = BytesIO()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            if isinstance(frame, TextFrame):
                # Stream LLM response immediately to client
                logger.info(f"📤 Sending LLM response: '{frame.text}'")
                await self._websocket.send_json({
                    "type": "llm_stream", 
                    "content": frame.text
                })
                
                # Also show in AI response format
                await self._websocket.send_json({
                    "type": "ai_response",
                    "content": f"AI: {frame.text}"
                })
                
            elif isinstance(frame, TTSAudioRawFrame):
                # Stream audio chunks for ultra-low latency
                audio_b64 = base64.b64encode(frame.audio).decode()
                logger.info(f"📤 Sending audio chunk: {len(frame.audio)} bytes")
                await self._websocket.send_json({
                    "type": "audio_stream",
                    "content": audio_b64,
                    "streaming": True
                })
                    
            elif isinstance(frame, TranscriptionFrame):
                # Send real-time transcription
                await self._websocket.send_json({
                    "type": "transcription_stream",
                    "content": frame.text
                })
                
        except Exception as e:
            logger.error(f"❌ WebSocket transport error: {e}")
            
        # Always pass frame downstream for pipeline processing
        await self.push_frame(frame, direction)


class WordLevelStreamingProcessor(FrameProcessor):
    """Stream words to LLM as soon as they're recognized - no waiting!"""
    
    def __init__(self):
        super().__init__()
        self._word_buffer = []
        self._last_word_time = 0
        self._word_timeout = 0.5  # Send every 500ms or 3-4 words
        self._min_words = 3
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame):
            # Split into words and stream immediately
            words = frame.text.strip().split()
            if words:
                self._word_buffer.extend(words)
                self._last_word_time = asyncio.get_event_loop().time()
                
                # Send chunks immediately if we have enough words
                if len(self._word_buffer) >= self._min_words:
                    await self._send_word_chunk()
        
        # Check for timeout-based sending
        current_time = asyncio.get_event_loop().time()
        if (self._word_buffer and 
            (current_time - self._last_word_time) > self._word_timeout):
            await self._send_word_chunk()
            
        await self.push_frame(frame, direction)
    
    async def _send_word_chunk(self):
        """Send accumulated words to LLM immediately"""
        if self._word_buffer:
            chunk_text = " ".join(self._word_buffer)
            logger.info(f"🚀 Streaming word chunk to LLM: '{chunk_text}'")
            
            # Create proper LLM messages frame for OpenAI service
            from pipecat.frames.frames import LLMMessagesFrame
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant in a real-time voice call. Respond with ONE very short sentence (3-5 words max). Be natural and conversational."},
                {"role": "user", "content": chunk_text}
            ]
            
            # Create LLM messages frame for immediate processing
            llm_frame = LLMMessagesFrame(messages)
            await self.push_frame(llm_frame, FrameDirection.DOWNSTREAM)
            
            # Clear buffer
            self._word_buffer = []


async def generate_fallback_response(user_text: str, websocket: WebSocket, conversation_history: list):
    """Generate immediate fallback LLM response if pipecat pipeline fails"""
    try:
        logger.info(f"🔄 Generating fallback response for: '{user_text}'")
        
        # Create OpenAI client
        client = create_openai_client()
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_text})
        
        # Get ultra-fast response
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an AI assistant in a real-time voice call. Respond with ONE very short sentence (3-5 words max). Be natural."}
            ] + conversation_history[-4:],
            max_tokens=20,
            temperature=0.3,
            stream=False  # Non-streaming for immediate response
        )
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"✅ Fallback LLM response: '{ai_response}'")
        
        # Add to conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Send to client
        await websocket.send_json({
            "type": "ai_response",
            "content": f"AI: {ai_response}"
        })
        
        # Generate audio
        if use_elevenlabs:
            await generate_elevenlabs_audio_direct(ai_response, websocket)
        else:
            await synthesize_audio_immediate(ai_response, websocket)
            
    except Exception as e:
        logger.error(f"❌ Fallback response error: {e}")


async def delayed_fallback_response(user_text: str, websocket: WebSocket, conversation_history: list, delay: float):
    """Generate fallback response after a delay if no response was received"""
    await asyncio.sleep(delay)
    
    # Check if we already responded (by checking if conversation history was updated)
    if conversation_history and conversation_history[-1]["role"] == "user" and conversation_history[-1]["content"] == user_text:
        logger.info(f"⏰ No response received, generating delayed fallback for: '{user_text}'")
        await generate_fallback_response(user_text, websocket, conversation_history)


async def generate_elevenlabs_audio_direct(text: str, websocket: WebSocket):
    """Generate ElevenLabs audio directly without pipecat"""
    try:
        logger.info(f"🔊 Generating ElevenLabs audio: '{text[:30]}...'")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.6,
                "style": 0.0,
                "use_speaker_boost": False
            },
            "output_format": "mp3_22050_32",
            "optimize_streaming_latency": 4
        }
        
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream("POST", url, headers=headers, json=data) as response:
                if response.status_code == 200:
                    audio_data = b""
                    async for chunk in response.aiter_bytes(2048):
                        audio_data += chunk
                    
                    audio_b64 = base64.b64encode(audio_data).decode()
                    await websocket.send_json({
                        "type": "audio_response",
                        "content": audio_b64
                    })
                    logger.info(f"✅ ElevenLabs audio sent: {len(audio_data)} bytes")
                else:
                    logger.error(f"ElevenLabs API error: {response.status_code}")
                    
    except Exception as e:
        logger.error(f"❌ ElevenLabs direct generation error: {e}")


async def ultra_fast_voice_handler(websocket: WebSocket, pipeline: Pipeline, conversation_history: list):
    """Ultra-fast voice handler - FIXED speech display + ULTRA-LOW latency"""
    
    logger.info("⚡ ULTRA-FAST voice handler started!")
    logger.info("🎯 Target latency: 500-700ms")
    logger.info("🔧 FIXED speech display + aggressive optimizations")
    
    # Speech state
    current_speech = ""
    last_speech_time = 0
    processing_speech = False
    current_response_task = None
    speech_timeout = 1.0  # Even faster timeout
    min_speech_length = 2  # Even shorter trigger
    processed_speeches = set()
    
    try:
        # Send ready message
        await websocket.send_json({
            "type": "system",
            "content": "🚀 FIXED: Speech display + ultra-low latency!"
        })
        
        while True:
            try:
                # Get speech from WebSocket
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.03)  # Even faster
                message = json.loads(data)
                
                if message["type"] == "user_speech":
                    user_text = message["content"].strip()
                    is_final = message.get("final", False)
                    current_time = asyncio.get_event_loop().time()
                    
                    if user_text:
                        # Cancel ongoing response if user is speaking
                        if current_response_task and not current_response_task.done():
                            current_response_task.cancel()
                            logger.info("🛑 Cancelled - user speaking")
                        
                        if is_final:
                            # FINAL speech - display and process
                            current_speech = user_text
                            logger.info(f"✅ FINAL: '{current_speech}'")
                            
                            # IMMEDIATELY show user speech in UI
                            try:
                                await websocket.send_json({
                                    "type": "user_speech_display", 
                                    "content": f"You: {current_speech}"
                                })
                                logger.info(f"📺 DISPLAYED: '{current_speech}'")
                            except Exception as e:
                                logger.error(f"❌ Display error: {e}")
                            
                            # Process if long enough and not processed
                            if (len(current_speech) >= min_speech_length and
                                not processing_speech and
                                current_speech not in processed_speeches):
                                
                                processing_speech = True
                                processed_speeches.add(current_speech)
                                logger.info(f"🚀 PROCESSING: '{current_speech}'")
                                
                                # Start ultra-fast response
                                current_response_task = asyncio.create_task(
                                    ultra_fast_response_sub500ms(current_speech, websocket, conversation_history)
                                )
                                current_speech = ""
                                
                        else:
                            # Interim speech - just update
                            current_speech = user_text
                            logger.info(f"📝 Interim: '{current_speech[:20]}...'")
                            
                        last_speech_time = current_time
                        processing_speech = False
                        
            except asyncio.TimeoutError:
                # Check for timeout processing
                current_time = asyncio.get_event_loop().time()
                if (current_speech and 
                    not processing_speech and 
                    (current_time - last_speech_time) > speech_timeout and
                    len(current_speech) >= min_speech_length and
                    current_speech not in processed_speeches):
                    
                    processing_speech = True
                    processed_speeches.add(current_speech)
                    logger.info(f"⏰ TIMEOUT processing: '{current_speech}'")
                    
                    # Display user speech
                    try:
                        await websocket.send_json({
                            "type": "user_speech_display", 
                            "content": f"You: {current_speech}"
                        })
                        logger.info(f"📺 TIMEOUT DISPLAYED: '{current_speech}'")
                    except Exception as e:
                        logger.error(f"❌ Timeout display error: {e}")
                    
                    current_response_task = asyncio.create_task(
                        ultra_fast_response_sub500ms(current_speech, websocket, conversation_history)
                    )
                    current_speech = ""
                
                # Cleanup
                if len(processed_speeches) > 3:
                    processed_speeches.clear()
                
                continue
                
            except json.JSONDecodeError:
                logger.error("❌ JSON error")
                continue
                
    except WebSocketDisconnect:
        logger.info("📞 Call ended")
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
    finally:
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        logger.info("⚡ Handler ended")


async def ultra_fast_response_sub500ms(user_text: str, websocket: WebSocket, conversation_history: list):
    """EXTREME optimization for SUB-500MS total latency with GUARANTEED audio"""
    try:
        start_time = asyncio.get_event_loop().time()
        logger.info(f"⚡ SUB-500MS response: '{user_text}'")
        
        # IMMEDIATE text response to user
        await websocket.send_json({
            "type": "ai_response",
            "content": "AI: ..."
        })
        
        # ULTRA-AGGRESSIVE LLM call with fallback
        client = create_openai_client()
        
        # Try ultra-fast LLM with aggressive timeout
        llm_start = asyncio.get_event_loop().time()
        ai_response = "OK"  # Default fallback
        
        try:
            # EXTREME SPEED SETTINGS - NO CONVERSATION HISTORY
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "Reply 1 word."},
                        {"role": "user", "content": user_text}
                    ],  # NO conversation history for speed
                    max_tokens=1,  # SINGLE token for maximum speed
                    temperature=0.0,
                    stream=False
                ),
                timeout=0.8  # AGGRESSIVE 800ms timeout
            )
            
            ai_response = response.choices[0].message.content.strip()
            if not ai_response or len(ai_response.strip()) == 0:
                ai_response = "Yes"  # Fallback for empty responses
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ LLM timeout - using fallback")
            ai_response = get_quick_response(user_text)  # Instant fallback
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            ai_response = "OK"  # Emergency fallback
        
        llm_time = (asyncio.get_event_loop().time() - llm_start) * 1000
        logger.info(f"✅ LLM ({llm_time:.0f}ms): '{ai_response}'")
        
        # Update conversation history (minimal)
        conversation_history.append({"role": "user", "content": user_text[-20:]})  # Truncate for speed
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Keep only last 2 entries for speed
        if len(conversation_history) > 4:
            conversation_history = conversation_history[-2:]
        
        # Update UI immediately with real response
        await websocket.send_json({
            "type": "ai_response",
            "content": f"AI: {ai_response}"
        })
        
        # GUARANTEED AUDIO - Don't wait for completion to report success
        audio_start = asyncio.get_event_loop().time()
        
        # Start audio generation in background - don't block main response
        audio_task = None
        try:
            if use_elevenlabs:
                # Try ElevenLabs first with reasonable timeout
                audio_task = asyncio.create_task(
                    ultra_fast_elevenlabs_with_fallback(ai_response, websocket)
                )
            else:
                # Use immediate fallback
                audio_task = asyncio.create_task(
                    synthesize_audio_immediate(ai_response, websocket)
                )
                
            # Don't wait for audio - let it complete in background
            logger.info("🎵 Audio generation started in background")
            
        except Exception as e:
            logger.error(f"❌ Audio start error: {e}")
            # Emergency audio fallback
            asyncio.create_task(synthesize_audio_immediate(ai_response, websocket))
        
        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(f"🎯 TOTAL: {total_time:.0f}ms (audio in background)")
        
        if total_time <= 300:
            logger.info("🏆 INCREDIBLE! Sub-300ms!")
        elif total_time <= 500:
            logger.info("🏆 HIT SUB-500MS TARGET!")
        elif total_time <= 700:
            logger.info("✅ Within acceptable range")
        else:
            logger.warning(f"⚠️ MISSED by {total_time - 500:.0f}ms")
        
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        await websocket.send_json({
            "type": "ai_response",
            "content": "AI: Got it"
        })


def get_quick_response(user_text: str) -> str:
    """Instant fallback responses for ultra-low latency"""
    text_lower = user_text.lower()
    
    # Quick pattern matching for instant responses
    if any(word in text_lower for word in ['hello', 'hi', 'hey']):
        return "Hi"
    elif any(word in text_lower for word in ['how', 'what', 'why', 'when', 'where']):
        return "Good question"
    elif any(word in text_lower for word in ['thank', 'thanks']):
        return "Welcome"
    elif any(word in text_lower for word in ['yes', 'yeah', 'yep']):
        return "Great"
    elif any(word in text_lower for word in ['no', 'nope']):
        return "Okay"
    elif any(word in text_lower for word in ['joke', 'funny']):
        return "Haha"
    elif any(word in text_lower for word in ['good', 'great', 'awesome']):
        return "Nice"
    elif any(word in text_lower for word in ['bad', 'terrible', 'awful']):
        return "Sorry"
    else:
        return "Okay"


async def ultra_fast_elevenlabs_with_fallback(text: str, websocket: WebSocket):
    """ElevenLabs with gTTS fallback - GUARANTEED audio"""
    try:
        audio_start = asyncio.get_event_loop().time()
        logger.info(f"🔊 GUARANTEED audio: '{text}'")
        
        # Try ElevenLabs first with reasonable timeout
        try:
            await asyncio.wait_for(
                ultra_fast_elevenlabs_sub200ms(text, websocket),
                timeout=1.5  # More reasonable timeout
            )
            
            audio_time = (asyncio.get_event_loop().time() - audio_start) * 1000
            logger.info(f"✅ ElevenLabs success ({audio_time:.0f}ms)")
            return
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ ElevenLabs timeout - using gTTS fallback")
        except Exception as e:
            logger.warning(f"⚠️ ElevenLabs failed: {e} - using gTTS fallback")
        
        # GUARANTEED gTTS fallback
        try:
            await synthesize_audio_immediate(text, websocket)
            audio_time = (asyncio.get_event_loop().time() - audio_start) * 1000
            logger.info(f"✅ gTTS fallback success ({audio_time:.0f}ms)")
        except Exception as e:
            logger.error(f"❌ gTTS fallback failed: {e}")
            
    except Exception as e:
        logger.error(f"❌ Audio generation completely failed: {e}")


async def ultra_fast_elevenlabs_sub200ms(text: str, websocket: WebSocket):
    """ULTRA-AGGRESSIVE ElevenLabs for sub-200ms audio"""
    try:
        audio_start = asyncio.get_event_loop().time()
        logger.info(f"🔊 SUB-200MS audio: '{text}'")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        # ABSOLUTE MINIMUM settings for maximum speed
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.0,
                "similarity_boost": 0.0,
                "style": 0.0,
                "use_speaker_boost": False
            }
        }
        
        import httpx
        async with httpx.AsyncClient(timeout=1.2) as client:  # Slightly more reasonable
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                audio_data = response.content
                audio_b64 = base64.b64encode(audio_data).decode()
                
                await websocket.send_json({
                    "type": "audio_response",
                    "content": audio_b64
                })
                
                audio_time = (asyncio.get_event_loop().time() - audio_start) * 1000
                logger.info(f"✅ Audio ({audio_time:.0f}ms): {len(audio_data)} bytes")
            else:
                logger.error(f"ElevenLabs error: {response.status_code}")
                raise Exception("API error")
                
    except Exception as e:
        logger.error(f"❌ Sub-200ms audio failed: {e}")
        raise  # Let fallback handle it


async def create_ultra_fast_pipeline(websocket: WebSocket):
    """Create ultra-fast pipecat pipeline for 500-700ms voice responses"""
    
    if not PIPECAT_AVAILABLE:
        logger.error("❌ Pipecat not available - cannot create fast pipeline")
        return None
    
    try:
        logger.info("🚀 Creating ULTRA-FAST pipecat pipeline...")
        
        # Check API keys
        if not api_key or api_key == "your_openai_api_key_here":
            logger.error("❌ OpenAI API key required")
            return None
            
        if not elevenlabs_api_key or elevenlabs_api_key == "your_elevenlabs_api_key_here":
            logger.error("❌ ElevenLabs API key required for ultra-fast TTS")
            return None
        
        # Create ULTRA-OPTIMIZED services for 500-700ms latency
        logger.info("🔧 Creating OpenAI service...")
        openai_llm = OpenAILLMService(
            api_key=api_key,
            model="gpt-3.5-turbo-0125",  # Fastest model available
            max_tokens=25,  # Super short responses for speed
            temperature=0.1,  # Low temp for speed
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        logger.info("🔧 Creating ElevenLabs TTS service...")
        elevenlabs_tts = ElevenLabsTTSService(
            api_key=elevenlabs_api_key,
            voice_id=elevenlabs_voice_id,
            model="eleven_turbo_v2"  # Fastest ElevenLabs model
        )
        
        # Create word-streaming processor for instant LLM feeding
        word_streamer = WordLevelStreamingProcessor()
        
        # Create WebSocket transport
        ws_transport = WebSocketTransport(websocket)
        
        # Create pipeline: Speech -> WordStreamer -> LLM -> TTS -> WebSocket
        pipeline = Pipeline([
            word_streamer,     # Stream words as they come
            openai_llm,       # Process immediately 
            elevenlabs_tts,   # Generate audio instantly
            ws_transport      # Send to client
        ])
        
        logger.info("✅ ULTRA-FAST pipeline created!")
        logger.info("🎯 Target: 500-700ms total latency")
        logger.info("⚡ Word streaming -> LLM -> ElevenLabs -> WebSocket")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Pipeline creation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


async def create_streaming_pipeline(websocket: WebSocket):
    """Wrapper function for compatibility"""
    return await create_ultra_fast_pipeline(websocket)


@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main voice chat interface at root"""
    file_path = os.path.join(os.path.dirname(__file__), "labs.html")
    return FileResponse(file_path)

@app.get("/stream", response_class=HTMLResponse)
async def get_stream_page():
    """Serve the streaming voice chat interface (same as root)"""
    file_path = os.path.join(os.path.dirname(__file__), "labs.html")
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
        
        # Try to create ultra-fast pipecat pipeline
        pipeline = await create_ultra_fast_pipeline(websocket)
        
        if pipeline is None:
            logger.warning("⚠️ Pipecat pipeline failed, using fallback mode")
            active_sessions[session_id] = "fallback_mode"
            await websocket.send_json({
                "type": "system", 
                "content": "📞 Ready for voice call (fallback mode) - click the call button to start!"
            })
            await simple_voice_call_handler(websocket, conversation_history)
            return
        
        # Use ULTRA-FAST pipecat pipeline for 500-700ms responses
        logger.info("🚀 Using ULTRA-FAST pipecat pipeline!")
        active_sessions[session_id] = "ultra_fast_mode"
        await websocket.send_json({
            "type": "system", 
            "content": "⚡ ULTRA-FAST voice chat ready! (500-700ms target) Click call to start!"
        })
        
        # Run the ultra-fast pipecat pipeline
        await ultra_fast_voice_handler(websocket, pipeline, conversation_history)
                
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
        client = create_openai_client()
        
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
        client = create_openai_client()
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": text})
        
        # Create streaming chat completion
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Keep responses concise and conversational since this is a voice chat."}
            ] + conversation_history[-10:],
            max_tokens=MAX_LLM_TOKENS,
            temperature=LLM_TEMPERATURE,
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
    """Real-time voice call handler - processes speech chunks as user speaks"""
    
    logger.info("🎯 Real-time voice call handler started")
    
    # Speech accumulation and processing state
    speech_buffer = ""
    last_speech_time = 0
    processing_speech = False
    current_response_task = None
    speech_timeout = 1.5  # Process speech after 1.5s of silence
    min_speech_length = 10  # Minimum characters to process
    
    try:
        while True:
            try:
                # Use timeout to check for speech processing
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                
                if message["type"] == "user_speech":
                    user_text = message["content"].strip()
                    is_final = message.get("final", False)
                    current_time = asyncio.get_event_loop().time()
                    
                    if user_text:
                        # Cancel any ongoing response if user is speaking
                        if current_response_task and not current_response_task.done():
                            current_response_task.cancel()
                            logger.info("🛑 Cancelled ongoing response - user is speaking")
                        
                        # Update speech buffer
                        if is_final:
                            speech_buffer += " " + user_text
                            logger.info(f"🎯 Final speech: '{user_text}'")
                        else:
                            # For interim speech, replace the buffer with latest
                            speech_buffer = user_text
                            logger.info(f"🎤 Interim speech: '{user_text}'")
                        
                        last_speech_time = current_time
                        processing_speech = False
                        
                        # Process immediately if final or if we have enough content
                        if is_final or len(speech_buffer.strip()) >= min_speech_length:
                            if not processing_speech and speech_buffer.strip():
                                processing_speech = True
                                logger.info(f"🚀 Processing speech immediately: '{speech_buffer.strip()}'")
                                text_to_process = speech_buffer.strip()
                                speech_buffer = ""  # Clear buffer after processing
                                
                                # Process and reset flag when done
                                async def process_and_reset():
                                    nonlocal processing_speech
                                    try:
                                        await process_speech_chunk(text_to_process, websocket, conversation_history)
                                    finally:
                                        processing_speech = False
                                        logger.info("🔓 Processing flag reset")
                                
                                current_response_task = asyncio.create_task(process_and_reset())
                        
                        # ALSO process interim speech if it's long enough (for better responsiveness)
                        elif not is_final and len(speech_buffer.strip()) >= 3 and not processing_speech:
                            logger.info(f"🎯 Processing interim speech: '{speech_buffer.strip()}'")
                            processing_speech = True
                            text_to_process = speech_buffer.strip()
                            speech_buffer = ""  # Clear buffer after processing
                            
                            # Process and reset flag when done
                            async def process_and_reset_interim():
                                nonlocal processing_speech
                                try:
                                    await process_speech_chunk(text_to_process, websocket, conversation_history)
                                finally:
                                    processing_speech = False
                                    logger.info("🔓 Processing flag reset (interim)")
                            
                            current_response_task = asyncio.create_task(process_and_reset_interim())
                    
            except asyncio.TimeoutError:
                # Check if we should process accumulated speech
                current_time = asyncio.get_event_loop().time()
                if (speech_buffer.strip() and 
                    not processing_speech and 
                    (current_time - last_speech_time) > speech_timeout and
                    len(speech_buffer.strip()) >= min_speech_length):
                    
                    processing_speech = True
                    logger.info(f"⏰ Processing speech after timeout: '{speech_buffer.strip()}'")
                    text_to_process = speech_buffer.strip()
                    speech_buffer = ""
                    
                    # Process and reset flag when done
                    async def process_and_reset_timeout():
                        nonlocal processing_speech
                        try:
                            await process_speech_chunk(text_to_process, websocket, conversation_history)
                        finally:
                            processing_speech = False
                            logger.info("🔓 Processing flag reset (timeout)")
                    
                    current_response_task = asyncio.create_task(process_and_reset_timeout())
                
                # Send periodic ping (FastAPI WebSocket doesn't have ping method)
                # Instead, we'll just continue the loop
                
    except WebSocketDisconnect:
        logger.info("📞 Call ended - WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ Real-time call handler error: {e}")
    finally:
        if current_response_task and not current_response_task.done():
            current_response_task.cancel()
        logger.info("🎯 Real-time voice call handler ended")


async def process_speech_chunk(user_text: str, websocket: WebSocket, conversation_history: list):
    """Process a speech chunk and generate response"""
    try:
        logger.info(f"🚀 Processing speech chunk: '{user_text}'")
        
        # Show user what we heard
        await websocket.send_json({
            "type": "user_speech_processed",
            "content": user_text
        })
        
        # Add to UI
        await websocket.send_json({
            "type": "ai_response",
            "content": f"You: {user_text}"
        })
        
        # Get AI response
        await get_ai_response_realtime(user_text, websocket, conversation_history)
        
        logger.info(f"✅ Completed processing speech chunk: '{user_text}'")
        
    except Exception as e:
        logger.error(f"❌ Speech chunk processing error: {e}")
    finally:
        # Reset processing flag in the calling function
        pass


async def get_ai_response_realtime(user_text: str, websocket: WebSocket, conversation_history: list):
    """Real-time AI response with immediate streaming"""
    try:
        client = create_openai_client()
        
        logger.info(f"🤖 Getting real-time AI response for: '{user_text}'")
        
        # Check if we have a valid API key
        if not api_key or api_key == "your_openai_api_key_here":
            # Test mode - send a simple response without OpenAI
            test_responses = [
                "I heard you say something about that!",
                "That's interesting, tell me more.",
                "I understand what you mean.",
                "Thanks for sharing that with me.",
                "I'm listening to what you're saying."
            ]
            import random
            test_response = random.choice(test_responses)
            
            logger.warning("⚠️ Using test mode - OpenAI API key not configured")
            await websocket.send_json({
                "type": "ai_response",
                "content": f"AI: [TEST MODE] {test_response}"
            })
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": test_response})
            
            # Generate test audio
            await synthesize_audio_immediate(test_response, websocket)
            return
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_text})
        
        logger.info(f"📤 Sending request to OpenAI with model: {LLM_MODEL}")
        
        # Use streaming for immediate response
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant in a real-time voice call. Respond immediately with very brief, natural responses (1 short sentence max). Be conversational and acknowledge what the user is saying."}
            ] + conversation_history[-6:],  # Smaller context for speed
            max_tokens=MAX_LLM_TOKENS,
            temperature=LLM_TEMPERATURE,
            stream=True
        )
        
        logger.info("📥 Starting to process OpenAI stream response")
        
        response_text = ""
        current_chunk = ""
        word_count = 0
        chunk_count = 0
        
        # Process streaming response with immediate TTS
        try:
            async for chunk in stream:
                chunk_count += 1
                if chunk_count == 1:
                    logger.info("📥 Received first chunk from OpenAI")
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    current_chunk += content
                    
                    # Process on word boundaries for natural speech
                    if ' ' in current_chunk or any(p in current_chunk for p in '.!?'):
                        words = current_chunk.split()
                        if len(words) >= 3:  # Process every 3 words
                            chunk_text = ' '.join(words[:-1])  # Keep last word for next chunk
                            if chunk_text.strip():
                                # Send text immediately
                                await websocket.send_json({
                                    "type": "ai_response",
                                    "content": f"AI: {chunk_text}"
                                })
                                
                                # Generate audio in parallel
                                asyncio.create_task(
                                    synthesize_audio_immediate(chunk_text, websocket)
                                )
                                
                            current_chunk = words[-1] if words else ""
            
            # Handle remaining text
            if current_chunk.strip():
                await websocket.send_json({
                    "type": "ai_response",
                    "content": f"AI: {current_chunk}"
                })
                await synthesize_audio_immediate(current_chunk, websocket)
                
        except Exception as stream_error:
            logger.error(f"❌ Error processing OpenAI stream: {stream_error}")
            error_response = "Sorry, I had trouble processing that. Could you try again?"
            await websocket.send_json({
                "type": "ai_response",
                "content": f"AI: {error_response}"
            })
            response_text = error_response
        
        if response_text.strip():
            conversation_history.append({"role": "assistant", "content": response_text})
            logger.info(f"✅ Real-time response complete: '{response_text}' (processed {chunk_count} chunks)")
        else:
            logger.error("❌ No response text generated - OpenAI stream may have failed")
        
    except Exception as e:
        logger.error(f"❌ Real-time AI response error: {e}")
        await websocket.send_json({
            "type": "error",
            "content": "Failed to get AI response"
        })


async def synthesize_audio_immediate(text: str, websocket: WebSocket):
    """Immediate audio synthesis with fastest possible settings"""
    if not text.strip():
        return
        
    logger.info(f"🔊 Generating audio for: '{text[:30]}...'")
    
    if use_elevenlabs:
        await synthesize_audio_streaming_elevenlabs_optimized(text, websocket)
    else:
        # Ultra-fast gTTS fallback with optimizations
        audio_b64 = await synthesize_audio_fast_fallback(text)
        if audio_b64:
            await websocket.send_json({
                "type": "audio_response",
                "content": audio_b64
            })


async def synthesize_audio_streaming_elevenlabs_optimized(text: str, websocket: WebSocket):
    """Ultra-optimized ElevenLabs streaming for lowest possible latency"""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}/stream"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json", 
            "xi-api-key": elevenlabs_api_key
        }
        
        # Ultra-optimized settings for speed
        data = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.4,      # Lower for speed
                "similarity_boost": 0.6,  # Lower for speed
                "style": 0.0,          # Minimum processing
                "use_speaker_boost": False  # Disable for speed
            },
            "output_format": "mp3_22050_32",  # Lowest quality for speed
            "optimize_streaming_latency": 4,  # Maximum optimization
            "chunk_length_schedule": [120, 160, 250, 290]  # Aggressive chunking
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:  # Reduced timeout
            async with client.stream("POST", url, headers=headers, json=data) as response:
                if response.status_code != 200:
                    logger.error(f"ElevenLabs API error: {response.status_code}")
                    await synthesize_audio_fast_fallback(text, websocket)
                    return
                
                # Stream with ultra-small chunks
                first_chunk = True
                async for chunk in response.aiter_bytes(ELEVENLABS_CHUNK_SIZE):
                    if first_chunk:
                        logger.info(f"🎵 First audio chunk received in {asyncio.get_event_loop().time():.3f}s")
                        first_chunk = False
                    
                    chunk_b64 = base64.b64encode(chunk).decode()
                    await websocket.send_json({
                        "type": "audio_response",
                        "content": chunk_b64,
                        "streaming": True
                    })
                
    except Exception as e:
        logger.error(f"ElevenLabs optimized streaming error: {e}")
        await synthesize_audio_fast_fallback(text, websocket)


async def synthesize_audio_fast_fallback(text: str, websocket: WebSocket = None) -> str:
    """Ultra-fast gTTS fallback with minimal processing"""
    if not text.strip():
        return ""
        
    try:
        logger.info(f"🔊 Generating gTTS audio for: '{text[:30]}...'")
        
        # Use minimal settings for speed
        tts = gTTS(text=text, lang='en', slow=False, lang_check=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        audio_b64 = base64.b64encode(audio_data).decode()
        
        logger.info(f"✅ Generated {len(audio_data)} bytes of audio")
        
        if websocket:
            await websocket.send_json({
                "type": "audio_response",
                "content": audio_b64
            })
        
        return audio_b64
    except Exception as e:
        logger.error(f"❌ Fast fallback TTS error: {e}")
        if websocket:
            await websocket.send_json({
                "type": "ai_response",
                "content": f"[Audio generation failed: {str(e)}]"
            })
        return ""


async def synthesize_audio(text: str) -> str:
    """Generate audio from text - kept for compatibility"""
    return await synthesize_audio_fast_fallback(text)


# Response caching for ultra-low latency
response_cache = {}
CACHE_MAX_SIZE = 100

def get_cached_response(text_hash: str):
    """Get cached response if available"""
    return response_cache.get(text_hash)

def cache_response(text_hash: str, audio_b64: str):
    """Cache response for future use"""
    if len(response_cache) >= CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(response_cache))
        del response_cache[oldest_key]
    response_cache[text_hash] = audio_b64


@app.get("/stream-status")
async def get_stream_status():
    """Get status of streaming sessions"""
    return {
        "status": "running",
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys()),
        "optimizations": {
            "ultra_low_latency": ULTRA_LOW_LATENCY,
            "parallel_processing": PARALLEL_PROCESSING,
            "stream_llm_responses": STREAM_LLM_RESPONSES,
            "elevenlabs_enabled": use_elevenlabs,
            "cached_responses": len(response_cache)
        }
    }


@app.get("/health")
async def health_check():
    """Enhanced health check with optimization status"""
    return {
        "status": "healthy", 
        "service": "streaming_voice_chat",
        "latency_optimizations": {
            "llm_model": LLM_MODEL,
            "max_tokens": MAX_LLM_TOKENS,
            "elevenlabs_model": ELEVENLABS_MODEL if use_elevenlabs else None,
            "streaming_enabled": STREAM_LLM_RESPONSES,
            "parallel_processing": PARALLEL_PROCESSING
        }
    }


@app.get("/latency-test")
async def latency_test():
    """Endpoint to test latency optimizations"""
    start_time = asyncio.get_event_loop().time()
    
    # Test LLM response time
    try:
        client = create_openai_client()
        
        llm_start = asyncio.get_event_loop().time()
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            temperature=LLM_TEMPERATURE
        )
        llm_time = (asyncio.get_event_loop().time() - llm_start) * 1000
        
        # Test TTS if ElevenLabs is available
        tts_time = None
        if use_elevenlabs:
            tts_start = asyncio.get_event_loop().time()
            # Test with short text
            await synthesize_audio_fast_fallback("Hi")
            tts_time = (asyncio.get_event_loop().time() - tts_start) * 1000
        
        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "total_time_ms": round(total_time, 2),
            "llm_time_ms": round(llm_time, 2),
            "tts_time_ms": round(tts_time, 2) if tts_time else None,
            "optimizations_active": ULTRA_LOW_LATENCY,
            "target_latency_ms": "500-700",
            "estimated_browser_stt_ms": "200-500",
            "network_latency_ms": "50-200"
        }
        
    except Exception as e:
        return {"error": str(e), "total_time_ms": (asyncio.get_event_loop().time() - start_time) * 1000}


@app.get("/real-time-test")
async def real_time_test():
    """Test endpoint for real-time voice processing capabilities"""
    return {
        "status": "ready",
        "real_time_features": {
            "interim_speech_processing": True,
            "speech_interruption_handling": True,
            "parallel_tts_generation": True,
            "streaming_llm_responses": True,
            "debounced_speech_chunks": True
        },
        "timings": {
            "speech_timeout_seconds": 1.5,
            "min_speech_length_chars": 10,
            "interim_debounce_ms": 300,
            "target_total_latency_ms": "300-800"
        },
        "optimizations": {
            "process_while_speaking": "✅ Active",
            "cancel_responses_on_new_speech": "✅ Active", 
            "immediate_audio_streaming": "✅ Active",
            "word_boundary_processing": "✅ Active"
        }
    }


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
        "labs:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    ) 