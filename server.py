import asyncio
import os
import re
import json
import uvicorn
import httpx
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- SETUP ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MURF_KEY = os.getenv("MURF_API_KEY")

if not OPENAI_KEY or not MURF_KEY:
    print("❌ ERROR: Missing API Keys")
    exit(1)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

# Config
MURF_URL = "https://global.api.murf.ai/v1/speech/stream"
MURF_VOICE_ID = "en-US-ken" 

async def stream_audio_from_murf(text: str, websocket: WebSocket):
    headers = { "api-key": MURF_KEY, "Content-Type": "application/json" }
    
    # 🔴 FIX: Changed model to "FALCON" (Gen2 is not supported on this URL)
    payload = {
        "text": text,
        "model": "FALCON",       # <--- CHANGED FROM GEN2 TO FALCON
        "voiceId": MURF_VOICE_ID,
        "format": "WAV",         
        "sampleRate": 24000,
        "multiNativeLocale": "en-US"
    }

    print(f"🎵 Requesting Audio for: {text[:20]}...")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", MURF_URL, json=payload, headers=headers) as response:
                
                if response.status_code != 200:
                    err = await response.aread()
                    print(f"⚠️ API Error {response.status_code}: {err}")
                    return

                is_first_chunk = True
                
                async for chunk in response.aiter_bytes():
                    if chunk:
                        # ✂️ STRIP WAV HEADER (First 44 bytes) to get Raw PCM
                        if is_first_chunk:
                            if len(chunk) > 44:
                                await websocket.send_bytes(chunk[44:])
                            is_first_chunk = False
                        else:
                            await websocket.send_bytes(chunk)
                            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

async def process_openai_stream(user_text: str, websocket: WebSocket):
    print(f"📩 Processing: {user_text}")
    try:
        stream = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Keep answers short (1 sentence)."},
                {"role": "user", "content": user_text}
            ],
            stream=True,
        )

        buffer = ""
        sentence_end_regex = re.compile(r'(?<=[.!?])\s+')

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content
                parts = sentence_end_regex.split(buffer)
                
                if len(parts) > 1:
                    sentence = parts[0].strip()
                    buffer = "".join(parts[1:])
                    if sentence:
                        await websocket.send_text(json.dumps({"type": "text", "content": sentence}))
                        await stream_audio_from_murf(sentence, websocket)

        if buffer.strip():
            await websocket.send_text(json.dumps({"type": "text", "content": buffer.strip()}))
            await stream_audio_from_murf(buffer.strip(), websocket)
            
        await websocket.send_text(json.dumps({"type": "status", "content": "done"}))

    except Exception as e:
        print(f"❌ OpenAI Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Connected")
    try:
        while True:
            data = await websocket.receive_text()
            await process_openai_stream(data, websocket)
    except WebSocketDisconnect:
        print("🔴 Disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)