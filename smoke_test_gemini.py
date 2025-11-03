import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
assert key, "no api key in .env"

genai.configure(api_key=key)
print("[info] google-generativeai version:", genai.__version__)

print("\n== MODELS supporting generateContent ==")
chat_models = []
for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print(" -", m.name)
        chat_models.append(m.name)

if not chat_models:
    raise SystemExit("no models for you")

model_name = chat_models[0]
clean_name = model_name.split("/")[-1]

print("\n[info] trying model:", clean_name)
model = genai.GenerativeModel(clean_name)
resp = model.generate_content("Reply with exactly: PONG")
print("[OK] response:", resp.text)
