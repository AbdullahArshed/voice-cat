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
    print("Silero VAD not available. Install with: pip install pipecat-ai[silero]")

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
        # Configure VAD if available
        vad_analyzer = SileroVADAnalyzer() if HAS_SILERO else None
        
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketTransport.Params(
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=HAS_SILERO,
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
        await runner.run(task)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = VoiceAgent()
    
    try:
        await agent.run_conversation(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
    <head><title>Voice Agent</title></head>
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
                    audioStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { sampleRate: 16000, channelCount: 1 } 
                    });
                    
                    ws = new WebSocket(`ws://${location.host}/ws`);
                    
                    ws.onopen = () => {
                        updateStatus('Connected - Speak now!');
                        setupAudio();
                    };
                    
                    ws.onmessage = (event) => {
                        if (event.data instanceof Blob) {
                            const audio = new Audio();
                            audio.src = URL.createObjectURL(event.data);
                            audio.play();
                        }
                    };
                    
                    ws.onclose = () => updateStatus('Disconnected');
                    ws.onerror = () => updateStatus('Error');
                    
                } catch (error) {
                    updateStatus('Failed: ' + error.message);
                }
            }

            function setupAudio() {
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
            }

            function disconnect() {
                if (ws) ws.close();
                if (audioStream) audioStream.getTracks().forEach(t => t.stop());
                if (audioContext) audioContext.close();
                updateStatus('Disconnected');
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)