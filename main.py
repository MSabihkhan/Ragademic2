from ingestion.loader import load_documents_to_chroma
from ingestion.pipeline import run_pipeline
from Vectorstore.index import buildChromaDB, get_index_from_chroma
from chat.engine import create_chat_engine
from config.settings import add_api_key
add_api_key("AIzaSyA-IU-fvl1lD7tZlI3Qn0PeXQwzTTu8BPY")
from google.genai.errors import ServerError
# Load vector index (assumes documents were indexed already via ChromaDB)

docs = load_documents_to_chroma()
print("Docs done")
nodes = run_pipeline(docs)
print("pipeline done")
index = buildChromaDB(nodes, "Theory-of-Automata")
print("✅ Indexing done.")
index = get_index_from_chroma("Theory-of-Automata")
print("Index Fetched")
# Create chat engine
chat_engine = create_chat_engine(index )
print("💬 Chat engine ready. Start asking questions.\n")

# Chat loop with Gemini 503 protection
while True:
    try:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break

        response = chat_engine.chat(user_input)
        print("Bot:", response.response)
        print("*************************")

    except ServerError as e:
        print("❌ Gemini LLM is overloaded (503). Please try again later.")
        print("*************************")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("*************************")
