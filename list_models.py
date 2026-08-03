from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

print("Available models jo generateContent support karte hain:\n")

for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"  {model.name}")