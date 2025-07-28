from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import tempfile
import os
import whisper
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

app = FastAPI(title="Voice Agent without Pipecat")

# Initialize Whisper model once (small model for demo)
model = whisper.load_model("small")

# Set up OpenAI client (new SDK usage)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def save_temp_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
    async with aiofiles.open(tmp_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return tmp_path


def transcribe_audio(file_path: str) -> str:
    result = model.transcribe(file_path)
    return result.get("text", "")


def generate_response(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # Make sure your key has access to this model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


@app.post("/voice-agent/")
async def voice_agent(file: UploadFile = File(...)):
    audio_path = await save_temp_file(file)
    try:
        transcript = transcribe_audio(audio_path)
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        response = generate_response(transcript)
        if not response:
            raise HTTPException(status_code=500, detail="LLM did not generate response")

        return JSONResponse(content={"transcript": transcript, "response": response})
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
