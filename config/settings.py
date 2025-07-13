from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from google.api_core.exceptions import ServiceUnavailable
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import time


llm = None
embed_model = None
# 2️⃣ Initialize Google LLM (Gemini)
def add_api_key(gemini_api_key):
    print ("Setting api key")
    os.environ['GOOGLE_API_KEY'] = gemini_api_key
    for attempt in range(5):
        try:
            print(f"🔁 Attempt {attempt+1}: Initializing Gemini Flash...")
            llm = GoogleGenAI(model="gemini-2.0-flash")
            #embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
            print("setting embed model")
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("Hugging face done")
            break
        except ServiceUnavailable as e:
            print(f"⚠️ Gemini Flash overloaded: {e}")
            time.sleep(5 * (2 ** attempt))
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    Settings.llm = llm
    Settings.embed_model = embed_model



# 4️⃣ Apply global Settings (optional)
Settings.llm = llm
Settings.embed_model = embed_model
