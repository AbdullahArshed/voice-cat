import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport

from langchain_core.messages import SystemMessage


class FrameEmitter(FrameProcessor):
    def __init__(self, initial_frame):
        super().__init__()
        self.initial_frame = initial_frame
        self._sent_initial = False

    def set_parent(self, parent):
        # Required by Pipecat processor chain
        self._parent = parent

    async def process(self, frame):
        if not self._sent_initial:
            self._sent_initial = True
            # Yield initial system message frame first
            yield self.initial_frame
        # Then yield incoming frame downstream
        yield frame


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


class DummyAudioSerializer:
    type = "audio/raw"

    def serialize(self, data):
        return data

    def deserialize(self, data):
        # Defensive check to prevent error if string received instead of bytes
        if isinstance(data, str):
            raise TypeError(
                "Expected bytes for audio data but received str. "
                "Ensure frontend sends raw binary audio, not JSON or text."
            )
        return data

    async def setup(self, frame):
        # No setup needed for dummy serializer
        pass


class TransportParams:
    def __init__(
        self,
        audio_out,
        audio_in_enabled,
        vad_analyzer,
        add_wav_header,
        vad_audio_passthrough=False,
        audio_in_sample_rate=16000,
        audio_in_filter=None,
        session_timeout=30,  # Default value
    ):
        self.audio_out = audio_out
        self.audio_in_enabled = audio_in_enabled
        self.vad_analyzer = vad_analyzer
        self.turn_analyzer = None
        self.add_wav_header = add_wav_header
        self.audio_in_passthrough = False
        self.camera_in_enabled = False
        self.serializer = DummyAudioSerializer()
        self._vad_audio_passthrough = vad_audio_passthrough
        self.audio_in_sample_rate = audio_in_sample_rate
        self.audio_in_filter = audio_in_filter
        self.session_timeout = session_timeout

    @property
    def vad_enabled(self):
        return self.audio_in_enabled

    @property
    def vad_audio_passthrough(self):
        return self._vad_audio_passthrough


class VoiceAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in .env file")
            raise ValueError("OPENAI_API_KEY is required")
        self.llm = OpenAILLMService(api_key=api_key, model="gpt-4o-mini")
        self.tts = OpenAITTSService(api_key=api_key, voice="alloy")
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            self.vad_analyzer = SileroVADAnalyzer()
            self.has_silero = True
        except ImportError:
            self.vad_analyzer = None
            self.has_silero = False
            logger.warning(
                "Silero VAD not available. Install with: pip install 'pipecat-ai[silero]'"
            )

    async def run_conversation(self, websocket: WebSocket):
        params = TransportParams(
            audio_out=True,
            audio_in_enabled=self.has_silero,
            vad_analyzer=self.vad_analyzer,
            add_wav_header=True,
            vad_audio_passthrough=True,
        )

        logger.debug(f"Initializing transport with params: {vars(params)}")
        transport = FastAPIWebsocketTransport(websocket=websocket, params=params)

        initial = LLMMessagesFrame(
            messages=[
                SystemMessage(
                    content="You are a helpful voice assistant. Keep responses short and conversational."
                )
            ]
        )
        context = FrameEmitter(initial)

        pipeline = Pipeline(
            [
                transport.input(),
                context,
                self.llm,
                self.tts,
                transport.output(),
            ]
        )
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        await runner.run(task)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.debug("WebSocket connection accepted")
    agent = VoiceAgent()
    try:
        await agent.run_conversation(websocket)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/")
async def index():
    return HTMLResponse(
        """
    <html>
    <head><title>Voice Agent</title><meta charset="UTF-8"></head>
    <body>
        <h2>🎙️ Voice Agent</h2>
        <div id="status">Disconnected</div>
        <button onclick="connect()">Connect</button>
        <button onclick="disconnect()">Disconnect</button>
        <script>
        let ws = null, audioStream = null, audioContext = null;

        function updateStatus(msg) {
            document.getElementById('status').textContent = msg;
        }

        async function connect() {
            try {
                if (!navigator.mediaDevices?.getUserMedia)
                    throw new Error('MediaDevices API not supported');
                audioStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1 }
                });
                ws = new WebSocket('ws://localhost:8000/ws');
                ws.binaryType = 'arraybuffer';  // Ensure binary messages are treated as ArrayBuffer
                ws.onopen = () => {
                    updateStatus('Connected - Speak now!');
                    setupAudio();
                };
                ws.onmessage = event => {
                    if (event.data instanceof Blob) {
                        const audio = new Audio();
                        audio.src = URL.createObjectURL(event.data);
                        audio.play().catch(console.error);
                    } else {
                        console.log('Non-audio message:', event.data);
                    }
                };
                ws.onclose = () => {
                    updateStatus('Disconnected');
                    disconnect();
                };
                ws.onerror = err => {
                    updateStatus('WebSocket Error');
                    console.error(err);
                };
            } catch (err) {
                updateStatus('Failed: ' + err.message);
                console.error(err);
            }
        }

        function floatTo16BitPCM(input) {
            const output = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
                output[i] = Math.max(-32768, Math.min(32767, input[i] * 32767));
            }
            return output;
        }

        function setupAudio() {
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(audioStream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = event => {
                if (ws?.readyState === WebSocket.OPEN) {
                    const audioData = event.inputBuffer.getChannelData(0);
                    const pcmData = floatTo16BitPCM(audioData);
                    // Send raw binary audio (ArrayBuffer)
                    ws.send(pcmData.buffer);
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
        }

        function disconnect() {
            ws?.close();
            audioStream?.getTracks().forEach(t => t.stop());
            audioContext?.close();
            ws = audioStream = audioContext = null;
            updateStatus('Disconnected');
        }
        </script>
    </body>
    </html>
    """
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)
