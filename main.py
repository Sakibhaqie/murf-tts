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
    print("❌ ERROR: Missing API Keys in .env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

# --- CONFIGURATION ---
MURF_URL = "https://global.api.murf.ai/v1/speech/stream"

# 1. UPDATED VOICE DB (Replaced Invalid 'Aarya' with 'Namrita')
VOICES_DB = {
    "en-US-ken":     {"name": "Ken",     "lang": "en-US", "label": "Ken (US Male)"},
    "en-US-natalie": {"name": "Natalie", "lang": "en-US", "label": "Natalie (US Female)"},
    "hi-IN-namrita": {"name": "Namrita", "lang": "hi-IN", "label": "Namrita (Hindi Female)"}, # ✅ Fixed Voice ID
    "en-UK-ryan":    {"name": "Ryan",    "lang": "en-UK", "label": "Ryan (UK Male)"},
    "en-AU-met":     {"name": "Met",     "lang": "en-AU", "label": "Met (AU Female)"}
}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/voices")
async def get_voices():
    # Convert DB to list format for Frontend
    voice_list = [{"id": k, "name": v["label"], "lang": v["lang"]} for k, v in VOICES_DB.items()]
    return {"voices": voice_list}

async def stream_audio_from_murf(text: str, voice_id: str, websocket: WebSocket):
    headers = { "api-key": MURF_KEY, "Content-Type": "application/json" }
    
    # 2. LOOKUP CORRECT LOCALE
    voice_info = VOICES_DB.get(voice_id, VOICES_DB["en-US-ken"])
    target_locale = voice_info["lang"]

    payload = {
        "text": text,
        "model": "FALCON",
        "voiceId": voice_id,
        "format": "WAV",         
        "sampleRate": 24000,
        "multiNativeLocale": target_locale 
    }

    print(f"🎵 Requesting Audio: '{text[:15]}...' | Voice: {voice_id} | Locale: {target_locale}")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", MURF_URL, json=payload, headers=headers) as response:
                
                if response.status_code != 200:
                    err = await response.aread()
                    print(f"⚠️ API Error {response.status_code}: {err.decode('utf-8')}")
                    return

                # Collect full WAV
                full_audio = b""
                async for chunk in response.aiter_bytes():
                    if chunk:
                        full_audio += chunk
                
                if len(full_audio) > 0:
                   await websocket.send_bytes(full_audio)
                            
    except Exception as e:
        print(f"❌ Murf Connection Error: {e}")

async def process_openai_stream(user_text: str, voice_id: str, websocket: WebSocket):
    print(f"📩 Processing: {user_text} (Voice: {voice_id})")

    # 3. ENHANCED SYSTEM PROMPT FOR LANGUAGE ENFORCEMENT
    voice_info = VOICES_DB.get(voice_id, VOICES_DB["en-US-ken"])
    voice_name = voice_info["name"]
    voice_lang = voice_info["lang"]

    # Explicitly tell GPT to speak Hindi if the voice is Hindi
    lang_instruction = "You must reply in HINDI (Devanagari script) or Hinglish." if "hi-IN" in voice_lang else "Reply in English."

    system_prompt = (
        f"You are a helpful voice assistant named {voice_name}. "
        f"Your voice settings are set to {voice_lang}. "
        f"{lang_instruction} "
        "Keep responses extremely concise (max 1 sentence) for real-time speech."
    )

    try:
        stream = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            stream=True,
        )

        buffer = ""
        sentence_end_regex = re.compile(r'(?<=[.!?|।])\s+') # Added Hindi Danda (।)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer += content
                
                parts = sentence_end_regex.split(buffer)
                
                if len(parts) > 1:
                    sentence = parts[0].strip()
                    buffer = "".join(parts[1:])
                    if sentence:
                        await websocket.send_text(json.dumps({"type": "text", "content": sentence}))
                        await stream_audio_from_murf(sentence, voice_id, websocket)

        if buffer.strip():
            await websocket.send_text(json.dumps({"type": "text", "content": buffer.strip()}))
            await stream_audio_from_murf(buffer.strip(), voice_id, websocket)
            
        await websocket.send_text(json.dumps({"type": "status", "content": "done"}))

    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        await websocket.send_text(json.dumps({"type": "text", "content": "Error generating response."}))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Client Connected")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                user_text = payload.get("text", "")
                voice_id = payload.get("voice_id", "en-US-ken")
                
                if user_text:
                    await process_openai_stream(user_text, voice_id, websocket)
            except json.JSONDecodeError:
                print("❌ Invalid JSON received")
                
    except WebSocketDisconnect:
        print("🔴 Client Disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)