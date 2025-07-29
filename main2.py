
import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport

# Try to import Silero VAD, fall back to None if not available
try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    HAS_SILERO = True
except ImportError:
    SileroVADAnalyzer = None
    HAS_SILERO = False
    logging.warning("Silero VAD not available. Install with: pip install 'pipecat-ai[silero]'")

logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI()

class VoiceAgent:
    def __init__(self):
        self.llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )
        
        self.tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="alloy"
        )
        
        self.context = OpenAILLMContext(
            messages=[{
                "role": "system", 
                "content": "You are a helpful voice assistant. Keep responses short and conversational."
            }]
        )
        
    async def run_conversation(self, websocket: WebSocket):
        # Configure VAD if available, otherwise disable it
        vad_analyzer = SileroVADAnalyzer() if HAS_SILERO else None
        vad_enabled = HAS_SILERO
        
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketTransport.Params(
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=vad_enabled,
                vad_analyzer=vad_analyzer,
                vad_audio_passthrough=True,
                serializer=FastAPIWebsocketTransport.JsonFrameSerializer()
            )
        )

        pipeline = Pipeline([
            transport.input(),
            self.context,
            self.llm,
            self.tts,
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        try:
            await runner.run(task)
        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = VoiceAgent()
    
    try:
        await agent.run_conversation(websocket)
    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")

@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
    <head>
        <title>Voice Agent</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h2>🎙️ Voice Agent</h2>
        <div id="status">Disconnected</div>
        <button onclick="connect()">Connect</button>
        <button onclick="disconnect()">Disconnect</button>
        
        <script>
            let ws = null;
            let audioStream = null;
            let audioContext = null;

            function updateStatus(msg) {
                document.getElementById('status').textContent = msg;
            }

            async function connect() {
                try {
                    // Log browser and context details for debugging
                    console.log('Browser:', navigator.userAgent);
                    console.log('Is Secure Context:', window.isSecureContext);
                    console.log('Location:', window.location.href);

                    // Check for MediaDevices API
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        throw new Error(
                            'MediaDevices API is not supported. ' +
                            'Ensure you are using a modern browser (Chrome, Firefox, Safari 15+) ' +
                            'and accessing the page via http://localhost:8000 (not 0.0.0.0).'
                        );
                    }
                    
                    // Request microphone access
                    audioStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { sampleRate: 16000, channelCount: 1 } 
                    });
                    
                    // Use explicit WebSocket URL for localhost
                    ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = () => {
                        updateStatus('Connected - Speak now!');
                        console.log('WebSocket connected');
                        setupAudio();
                    };
                    
                    ws.onmessage = (event) => {
                        if (event.data instanceof Blob) {
                            console.log('Received audio response');
                            const audio = new Audio();
                            audio.src = URL.createObjectURL(event.data);
                            audio.play().catch(err => console.error('Audio playback error:', err));
                        }
                    };
                    
                    ws.onclose = () => {
                        updateStatus('Disconnected');
                        console.log('WebSocket disconnected');
                        disconnect();
                    };
                    
                    ws.onerror = (error) => {
                        updateStatus('WebSocket Error');
                        console.error('WebSocket error:', error);
                    };
                    
                } catch (error) {
                    updateStatus('Failed: ' + error.message);
                    console.error('Connection error:', error);
                }
            }

            function setupAudio() {
                try {
                    audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(audioStream);
                    const processor = audioContext.createScriptProcessor(4096, 1, 1);
                    
                    processor.onaudioprocess = (event) => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            const audioData = event.inputBuffer.getChannelData(0);
                            const int16Data = new Int16Array(audioData.length);
                            for (let i = 0; i < audioData.length; i++) {
                                int16Data[i] = audioData[i] * 32767;
                            }
                            ws.send(int16Data.buffer);
                        }
                    };
                    
                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    console.log('Audio processing setup complete');
                } catch (error) {
                    updateStatus('Audio setup error: ' + error.message);
                    console.error('Audio setup error:', error);
                }
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
                if (audioStream) {
                    audioStream.getTracks().forEach(t => t.stop());
                    audioStream = null;
                }
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                updateStatus('Disconnected');
                console.log('Disconnected and cleaned up');
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)
