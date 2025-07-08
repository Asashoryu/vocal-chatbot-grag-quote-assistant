# --- Global Parameters (can be adjusted) ---
RECORDING_DURATION_SECONDS = 3  # Default duration for new recordings
SAMPLE_RATE = 16000  # This is the target sample rate for ASR and Speaker Recognition models
# NEW: Common sample rate for TTS models (e.g., HiFi-GAN)
TTS_SAMPLE_RATE = 44100  # 22050
AUDIO_RECORDS_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/audio_recordings"
# This is where "registered" embeddings will be saved
EMBEDDING_OUTPUT_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/speaker_embeddings"
# A common threshold for speaker verification (adjust as needed)
FASTPITCH_EMBEDDINGS_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/fms_embeddings_192"
# A common threshold for speaker verification (adjust as needed)
VERIFICATION_THRESHOLD = 0.4

# NEW: Flag to control TTS for computer answers
USE_TTS_FOR_ANSWERS = True


# --- Neo4j Connection and Search Functions ---
# IMPORTANT: Replace these with your actual Neo4j connection details.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "amonolexandr"  # <--- REPLACE WITH YOUR ACTUAL NEO4J PASSWORD

# Mapping of user-friendly entity types to their corresponding Full-Text Index names
# This is a constant, so it can stay global or be passed in if you want more flexibility
ENTITY_INDEX_MAP = {
    "1": {"name": "Author", "index": "authorNamesIndex"},
    "2": {"name": "Quote", "index": "quoteTextsIndex"},
    "3": {"name": "Context", "index": "contextTextsIndex"},
    "4": {"name": "Detail", "index": "detailTextsIndex"},
    "5": {"name": "All Content", "index": "allTextContentIndex"}
}
