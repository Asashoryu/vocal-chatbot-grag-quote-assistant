# --- 1. Colab Setup and Dependency Installation ---
# This cell installs all necessary libraries. It might take a few minutes.

# Enable GPU runtime: Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU

print("Installing NeMo and other dependencies...")

# Install NeMo toolkit (ASR and Speaker Recognition components)
# Use 'main' branch for latest features or 'stable' for more stability
!python - m pip install git+[https://github.com/NVIDIA/NeMo.git@main  # egg=nemo_toolkit](https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit)[asr]
!pip install numpy scipy scikit-learn neo4j soundfile

# For audio processing (if not already part of nemo_toolkit[asr])
!pip install librosa torchaudio

print("Installation complete. Restarting runtime (if prompted by Colab)...")
# If Colab prompts to restart runtime after installation, please do so.

# After restarting, re-run cells from here.
```

```python
# --- 2. Imports and Configuration ---

import os
import numpy as np
import json
import re
import asyncio
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf  # For reading/writing audio files
from IPython.display import Audio, display  # For playing audio in Colab
import torch  # NeMo models are PyTorch-based
# For ASR and Speaker Recognition models
import nemo.collections.asr as nemo_asr
# import nemo.collections.tts as nemo_tts # For TTS models (if you integrate a real one)

print("All necessary libraries imported.")

# --- Configuration ---
# IMPORTANT: Replace these with your actual Neo4j connection details.
# For Colab, you'd typically use a cloud-hosted Neo4j (e.g., AuraDB)
# or run a local instance with port forwarding. For this demo,
# the Neo4j part will be simulated if connection fails.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password") # <<< IMPORTANT: REPLACE OR USE ENV VAR

# --- ASR Module Configuration ---
# Using a placeholder for ASR model. You can replace with a real ASR model from NeMo or HuggingFace.
ASR_MODEL_NAME = "simulated_asr_model" # "nvidia/stt_en_conformer_ctc_large" for a real NeMo ASR

# --- Speaker Identification Module Configuration ---
# Using TitaNet-Large for speaker embeddings
SPEAKER_ID_MODEL_NAME = "titanet_large" # Pre-trained TitaNet model from NeMo
VOICE_EMBEDDINGS_FILE = "voice_embeddings.json" # File to store/load voice embeddings
SPEAKER_RECOGNITION_THRESHOLD = 0.75 # Adjust this threshold based on your data and model performance

# --- Chatbot Module Configuration ---
# Placeholder for Gemini API key (if using Gemini for NL generation)
# For Colab, you'd usually set this as a Colab Secret or directly in the code for testing.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") # <<< IMPORTANT: REPLACE WITH YOUR ACTUAL GEMINI API KEY

# --- TTS Module Configuration ---
# Using a placeholder for TTS model. You can replace with a real TTS model from NeMo.
TTS_MODEL_NAME = "simulated_tts_model" # "nvidia/tts_en_fastpitch" for a real NeMo TTS
DEFAULT_TTS_VOICE_PREFERENCE = "default_female_voice"

# Example user voice preferences (in a real system, these would be stored persistently)
# This could map user IDs to specific voice IDs or style parameters for your TTS model
USER_VOICE_PREFERENCES = {
    "user_123": "male_voice_1",
    "user_456": "female_voice_2",
    "unidentified": DEFAULT_TTS_VOICE_PREFERENCE
}

# Mapping of user-friendly entity types to their corresponding Full-Text Index names
# This is used by the chatbot to select the correct Neo4j index
ENTITY_INDEX_MAP = {
    "Author": {"index": "authorNamesIndex", "label_prop": "name"},
    "Quote": {"index": "quoteTextsIndex", "label_prop": "text"},
    "Context": {"index": "contextTextsIndex", "label_prop": "text"},
    "Detail": {"index": "detailTextsIndex", "label_prop": "text"},
    # 'name' for Author, 'text' for others
    "All Content": {"index": "allTextContentIndex", "label_prop": "text"}
}

# Create an empty voice_embeddings.json file if it doesn't exist
if not os.path.exists(VOICE_EMBEDDINGS_FILE):
    with open(VOICE_EMBEDDINGS_FILE, 'w') as f:
        json.dump({}, f)
    print(f"Created empty {VOICE_EMBEDDINGS_FILE}")
```

```python
# --- 3. Module Definitions ---

# --- 3.1 Neo4j Utilities (Placeholder) ---
class Neo4jConnector:
    """
    Manages the connection to the Neo4j database.
    Encapsulates driver initialization and closing.
    NOTE: For Colab, a live connection requires a publicly accessible Neo4j instance.
    """
    def __init__(self, uri, username, password):
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None
        self._connected = False

    def connect(self):
        """Initializes the Neo4j driver."""
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
            self._driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
            self._connected = True
            return True
        except ServiceUnavailable as e:
            print(
                f"Connection failed: Neo4j database is not running or accessible at {self._uri}. Error: {e}")
            print("Neo4j search will be simulated.")
            self._connected = False
            return False
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print(
                "Please check your URI, username, and password. Neo4j search will be simulated.")
            self._connected = False
            return False

    def close(self):
        """Closes the Neo4j driver."""
        if self._driver and self._connected:
            self._driver.close()
            print("Neo4j driver closed.")

    def get_driver(self):
        """Returns the initialized Neo4j driver."""
        return self._driver

    def is_connected(self):
        """Returns True if connected to Neo4j, False otherwise."""
        return self._connected

def search_entity(driver, entity_type: str, substring: str, k: int=10, is_neo4j_connected: bool=False):
    """
    Performs a full-text search on the selected entity type.
    Simulates search if Neo4j is not connected.
    """
    if not is_neo4j_connected:
        print(
            f"(Simulated Neo4j Search) Searching for '{substring}' in {entity_type}...")
        # Simulate some results for demonstration
        simulated_results = []
        if "albert" in substring.lower() and entity_type == "Author":
            simulated_results.append(
                {"NodeType": "Author", "Content": "Albert Einstein", "Score": 0.95})
            simulated_results.append(
                {"NodeType": "Author", "Content": "Albert Camus", "Score": 0.88})
        elif "life" in substring.lower() and entity_type == "Quote":
            simulated_results.append(
                {"NodeType": "Quote", "Content": "Life is what happens when you're busy making other plans.", "Score": 0.92})
        elif "all" in entity_type.lower() and "wisdom" in substring.lower():
             simulated_results.append(
                 {"NodeType": "Quote", "Content": "The only true wisdom is in knowing you know nothing.", "Score": 0.90})
        else:
            simulated_results.append(
                {"NodeType": entity_type, "Content": f"Simulated result for '{substring}'", "Score": 0.7})
        return simulated_results[:k]

    # Actual Neo4j search logic (only runs if connected)
    entity_info = ENTITY_INDEX_MAP.get(entity_type)
    if not entity_info:
        print(
            f"Invalid entity type: {entity_type}. Please choose from {list(ENTITY_INDEX_MAP.keys())}.")
        return []

    index_name = entity_info["index"]
    results = []

    with driver.session() as session:
        search_term = f"{substring}*"

        query = f"""
        CALL db.index.fulltext.queryNodes('{index_name}', $searchTerm) YIELD node, score
        RETURN labels(node) AS NodeType,
               CASE
                   WHEN 'Author' IN labels(node) THEN node.name
                   WHEN 'Quote' IN labels(node) THEN node.text
                   WHEN 'Context' IN labels(node) THEN node.text
                   WHEN 'Detail' IN labels(node) THEN node.text
                   ELSE 'N/A'
               END AS Content,
               score
        ORDER BY score DESC
        LIMIT $k
        """

        try:
            result = session.run(query, searchTerm=search_term, k=k)
            for record in result:
                results.append({
                    "NodeType": ", ".join(record["NodeType"]),
                    "Content": record["Content"],
                    "Score": record["score"]
                })
        except Exception as e:
            print(
                f"An error occurred during the search for {entity_type}: {e}")
            print(
                f"Please ensure the full-text index '{index_name}' exists and is online.")
    return results

# --- 3.2 ASR Module (Simulated) ---
class ASRModule:
    """
    Automatic Speech Recognition (ASR) module.
    Uses pre-trained models to transcribe voice commands.
    NOTE: This is simulated for Colab. You can integrate a real ASR model here.
    """
    def __init__(self):
        # Placeholder for loading a real ASR model
        # self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=ASR_MODEL_NAME)
        print(f"ASR Module initialized. Model: {ASR_MODEL_NAME}")

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribes audio data into text.
        For Colab demo, it prompts for text input.
        """
        print("\n(ASR) Please type your voice command (e.g., 'search author Albert Einstein'):")
        transcribed_text = input("You: ").strip()
        return transcribed_text

# --- 3.3 Speaker Identification Module (TitaNet Integration) ---
class SpeakerIDModule:
    """
    Speaker Identification module using NVIDIA NeMo's TitaNet.
    Enables user registration and recognition based on voice embeddings.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SpeakerIDModule using device: {self.device}")
        self.model = self._load_speaker_id_model(SPEAKER_ID_MODEL_NAME)
        self.user_embeddings = self._load_embeddings() # Load registered user embeddings
        print(
            f"Speaker Identification Module initialized with model: {SPEAKER_ID_MODEL_NAME}")

    def _load_speaker_id_model(self, model_name):
        """
        Loads a pre-trained Speaker ID model (TitaNet) from NeMo.
        """
        print(f"Loading Speaker ID model: {model_name} from NeMo NGC...")
        try:
            # Load TitaNet model
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
            model.eval()  # Set to evaluation mode
            model.to(self.device)  # Move model to GPU if available
            return model
        except Exception as e:
            print(f"Error loading NeMo Speaker ID model: {e}")
            print(
                "Please ensure you have a GPU runtime enabled and NeMo is correctly installed.")
            print("Simulating Speaker ID model.")
            return "Simulated Speaker ID Model"

    def _generate_embedding(self, audio_path: str) -> np.ndarray:
        """
        Generates a voice embedding from an audio file path using the loaded TitaNet model.
        """
        if self.model == "Simulated Speaker ID Model":
            print("(Simulated Speaker ID) Generating dummy embedding...")
            return np.random.rand(192)  # Simulate Titanet's 192-dim embedding

        try:
            # NeMo's get_embedding expects a file path.
            # Make sure audio_path points to a valid 16kHz mono WAV file.
            embedding = self.model.get_embedding(audio_path=audio_path)
            return embedding.cpu().numpy()  # Move to CPU and convert to numpy
        except Exception as e:
            print(f"Error generating embedding from {audio_path}: {e}")
            print("Ensure audio is 16kHz mono WAV. Returning dummy embedding.")
            return np.random.rand(192)

    def _load_embeddings(self) -> dict:
        """
        Loads registered user voice embeddings from a JSON file.
        """
        if os.path.exists(VOICE_EMBEDDINGS_FILE):
            with open(VOICE_EMBEDDINGS_FILE, 'r') as f:
                data = json.load(f)
                # Convert list back to numpy array
                return {user_id: np.array(embedding) for user_id, embedding in data.items()}
        return {}

    def _save_embeddings(self):
        """
        Saves registered user voice embeddings to a JSON file.
        """
        serializable_embeddings = {user_id: embedding.tolist() for user_id, embedding in self.user_embeddings.items()}
        with open(VOICE_EMBEDDINGS_FILE, 'w') as f:
            json.dump(serializable_embeddings, f, indent=4)

    def register_user(self, user_id: str, audio_path: str) -> bool:
        """
        Registers a new user by generating and storing their voice embedding.

        Args:
            user_id (str): Unique identifier for the user.
            audio_path (str): Path to the audio file for the user's voice.

        Returns:
            bool: True if registration is successful, False otherwise.
        """
        if user_id in self.user_embeddings:
            print(f"User '{user_id}' already registered.")
            return False

        print(f"Registering user '{user_id}' from audio: {audio_path}...")
        embedding = self._generate_embedding(audio_path)
        if embedding is None:
            print(
                f"Failed to generate embedding for user '{user_id}'. Registration failed.")
            return False

        self.user_embeddings[user_id] = embedding
        self._save_embeddings()
        print(f"User '{user_id}' registered successfully.")
        return True

    def recognize_speaker(self, audio_path: str) -> str:
        """
        Recognizes the speaker from an audio file.

        Args:
            audio_path (str): Path to the audio file of the speaker.

        Returns:
            str: The recognized user_id, or "unidentified" if no match.
        """
        if not self.user_embeddings:
            print("No users registered for speaker identification.")
            return "unidentified"

        print(f"Recognizing speaker from audio: {audio_path}...")
        query_embedding = self._generate_embedding(audio_path)
        if query_embedding is None:
            print("Failed to generate query embedding. Speaker unidentified.")
            return "unidentified"

        best_match_id = "unidentified"
        highest_score = SPEAKER_RECOGNITION_THRESHOLD # Only consider scores above threshold

        for user_id, registered_embedding in self.user_embeddings.items():
            score = cosine_similarity(query_embedding.reshape(1, -1), registered_embedding.reshape(1, -1))[0][0]

            print(f"  Comparing with {user_id}: Score = {score:.2f}")
            if score > highest_score:
                highest_score = score
                best_match_id = user_id

        if best_match_id != "unidentified":
            print(
                f"Speaker recognized as '{best_match_id}' (Score: {highest_score:.2f})")
        else:
            print("Speaker unidentified.")

        return best_match_id

# --- 3.4 Chatbot Module ---
async def call_gemini_api(prompt: str) -> str:
    """
    Placeholder for making a call to the Gemini API for text generation.
    NOTE: Replace with actual Gemini API integration.
    """
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY:
        print("\n(Simulated LLM Call) Gemini API Key not configured. Simulating response.")
        if "search" in prompt.lower() and "results" in prompt.lower():
            return f"Here are the simulated search results for your query: {prompt.split('results: ')[1].split('Based on')[0].strip()}"
        elif "hello" in prompt.lower():
            return "Hello there! How can I help you today with quotes?"
        else:
            return f"I received your request: '{prompt}'. (Simulated LLM response)"

    # Actual Gemini API integration (requires google-generativeai library)
    # from google.generativeai import GenerativeModel, configure
    # configure(api_key=GEMINI_API_KEY)
    # model = GenerativeModel('gemini-2.0-flash') # Or other models
    # try:
    #     response = await model.generate_content_async(prompt)
    #     return response.text
    # except Exception as e:
    #     print(f"Error calling Gemini API: {e}")
    #     return "I'm sorry, I couldn't generate a response at the moment."


class ChatbotModule:
    """
    Chatbot module: a conversational interface based on the Wikiquote graph.
    """
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector
        self.driver = neo4j_connector.get_driver()
        print("Chatbot Module initialized.")

    async def generate_response(self, user_id: str, command_text: str) -> str:
        """
        Generates a natural language response based on the user's command
        and the Wikiquote graph data.
        """
        command_text_lower = command_text.lower()
        response = ""

        # --- Command Parsing Logic (Simple Regex-based) ---
        # Example commands:
        # "search author 'Albert Einstein'"
        # "find quote 'happiness'"
        # "search context 'philosophy'"
        # "get details for 'book chapter 1'"
        # "search all content 'life is'"

        match_author = re.search(r"search author ['\"](.+?)['\"]", command_text_lower)
        match_quote = re.search(r"find quote ['\"](.+?)['\"]", command_text_lower)
        match_context = re.search(r"search context ['\"](.+?)['\"]", command_text_lower)
        match_detail = re.search(r"get details for ['\"](.+?)['\"]", command_text_lower)
        match_all_content = re.search(r"search all content ['\"](.+?)['\"]", command_text_lower)

        search_results = []
        entity_type = None
        search_query = None

        if match_author:
            entity_type = "Author"
            search_query = match_author.group(1)
        elif match_quote:
            entity_type = "Quote"
            search_query = match_quote.group(1)
        elif match_context:
            entity_type = "Context"
            search_query = match_context.group(1)
        elif match_detail:
            entity_type = "Detail"
            search_query = match_detail.group(1)
        elif match_all_content:
            entity_type = "All Content"
            search_query = match_all_content.group(1)

        if entity_type and search_query:
            print(
                f"Chatbot: Detected command to search {entity_type} for '{search_query}'")
            # Pass Neo4j connection status to search_entity
            search_results = search_entity(self.driver, entity_type, search_query, k=5, is_neo4j_connected=self.neo4j_connector.is_connected())

            if search_results:
                formatted_results = []
                for i, res in enumerate(search_results):
                    formatted_results.append(
                        f"{i+1}. Type: {res['NodeType']}, Content: '{res['Content']}' (Score: {res['Score']:.2f})")

                llm_prompt = f"The user asked to search for '{search_query}' in {entity_type}. Here are the top results from the Wikiquote graph:\n" + "\n".join(formatted_results) + "\n\nBased on these results, provide a concise and helpful natural language response to the user. Do not explicitly mention 'score' unless asked."
                response = await call_gemini_api(llm_prompt)
            else:
                response = await call_gemini_api(f"I couldn't find any {entity_type} matching '{search_query}' in the Wikiquote graph. Is there anything else I can help you with?")
        elif "hello" in command_text_lower or "hi" in command_text_lower:
            response = await call_gemini_api("Hello! How can I assist you with quotes today?")
        else:
            response = await call_gemini_api(f"I'm not sure how to process '{command_text}'. You can try commands like 'search author \"Albert Einstein\"' or 'find quote \"happiness\"'.")

        return response

# --- 3.5 TTS Module (Simulated) ---
class TTSModule:
    """
    Personalized Text-to-Speech (TTS) module.
    Uses pre-trained models to generate vocals, configured by user voice preferences.
    NOTE: This is simulated for Colab. You can integrate a real TTS model here.
    """
    def __init__(self):
        # Placeholder for loading a real TTS model
        # self.model = nemo_tts.models.FastPitchModel.from_pretrained(model_name=TTS_MODEL_NAME)
        print(f"TTS Module initialized. Model: {TTS_MODEL_NAME}")

    def synthesize_speech(self, text: str, user_id: str) -> np.ndarray:
        """
        Synthesizes speech from text, applying personalized voice preferences.
        For Colab demo, it prints and plays a dummy audio.
        """
        voice_preference = USER_VOICE_PREFERENCES.get(user_id, DEFAULT_TTS_VOICE_PREFERENCE)

        print(
            f"\n(TTS) Synthesizing text for '{user_id}' with voice '{voice_preference}': '{text}'")

        # Simulate audio generation (replace with real TTS model inference)
        # In a real system, you'd generate audio using the loaded model and voice preference
        # e.g., audio_data = self.model.generate_audio(text, speaker=voice_preference)

        # For demonstration, generate a dummy audio array
        sample_rate = 16000
        duration_seconds = max(1, len(text.split()) * 0.2) # Longer text = longer dummy audio
        dummy_audio_data = np.random.uniform(-0.5, 0.5, int(duration_seconds * sample_rate)).astype(np.float32)

        # Display and play audio in Colab
        display(Audio(dummy_audio_data, rate=sample_rate, autoplay=True))

        return dummy_audio_data

# --- 4. Main Orchestration Function ---
async def main_vocal_chatbot_system():
    """
    Main function to run the interactive vocal chatbot system.
    Orchestrates ASR, Speaker ID, Chatbot, and TTS modules.
    """
    print("--- Initializing Vocal Chatbot System ---")

    # 1. Initialize Neo4j Connector
    neo4j_connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    neo4j_connector.connect()  # Attempt connection, will print status

    # 2. Initialize ASR Module
    asr_module = ASRModule()

    # 3. Initialize Speaker Identification Module
    speaker_id_module = SpeakerIDModule()

    # 4. Initialize Chatbot Module
    chatbot_module = ChatbotModule(neo4j_connector)

    # 5. Initialize TTS Module
    tts_module = TTSModule()

    print("\n--- System Ready ---")
    print("Type 'register' to register a new user with an audio file.")
    print("Type 'exit' to quit.")
    print("For voice commands, you will be prompted to type the transcription.")

    try:
        # --- Main Interaction Loop ---
        while True:
            print("\n----------------------------------------")
            print("Awaiting command...")

            # --- Simulate Audio Input ---
            # For a real system, you'd capture live audio here.
            # For Colab, we'll simulate by asking for a path to a WAV file
            # or allow typing for ASR.

            command_type = input("Enter 'audio' to process an audio file, 'text' to type command, or 'register' to register: ").strip().lower()

            audio_path_for_asr = None
            if command_type == 'audio':
                audio_path_for_asr = input("Enter path to a 16kHz mono WAV file (e.g., /content/my_audio.wav): ").strip()
                if not os.path.exists(audio_path_for_asr):
                    print(
                        "File not found. Please upload a WAV file or provide a valid path.")
                    continue
                # For demonstration, we'll still use text input for ASR,
                # but the audio_path will be used for Speaker ID.
                # In a real ASR integration, you'd pass the actual audio data.
                transcribed_text = asr_module.transcribe_audio(np.array([])) # Dummy audio data for ASR sim
            elif command_type == 'text':
                transcribed_text = asr_module.transcribe_audio(np.array([])) # Dummy audio data for ASR sim
            elif command_type == 'register':
                new_user_id = input("Enter new user ID to register: ").strip()
                if not new_user_id:
                    print("User ID cannot be empty.")
                    continue
                reg_audio_path = input(f"Enter path to 16kHz mono WAV file for '{new_user_id}' registration: ").strip()
                if not os.path.exists(reg_audio_path):
                    print(
                        "File not found. Please upload a WAV file or provide a valid path.")
                    continue
                speaker_id_module.register_user(new_user_id, reg_audio_path)
                continue
            elif command_type == 'exit':
                print("Exiting system.")
                break
            else:
                print(
                    "Invalid command type. Please choose 'audio', 'text', 'register', or 'exit'.")
                continue

            # If a valid audio path was provided, use it for speaker ID
            # Otherwise, for 'text' commands, we'll simulate an 'unidentified' speaker or use a default.
            speaker_audio_path = audio_path_for_asr if audio_path_for_asr else None

            # Speaker Identification: Recognize the speaker
            if speaker_audio_path:
                user_id = speaker_id_module.recognize_speaker(speaker_audio_path)
            else:
                user_id = "unidentified" # Default for text-only commands
                print(
                    "No audio provided for speaker recognition. Assuming 'unidentified' user.")

            # Chatbot: Generate response based on command and user ID
            response_text = await chatbot_module.generate_response(user_id, transcribed_text)
            print(f"Chatbot Response: '{response_text}'")

            # TTS: Synthesize personalized speech
            tts_module.synthesize_speech(response_text, user_id)

            print("Response delivered.")

    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up resources
        neo4j_connector.close()
        print("System shutdown complete.")

if __name__ == "__main__":
    # Ensure asyncio is compatible with Jupyter/Colab environments
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If a loop is already running (e.g., in Jupyter/Colab), schedule the task
        task = loop.create_task(main_vocal_chatbot_system())
        # You might need to wait for the task to complete if running other async code
        # await task
    else:
        # Run the asynchronous main function
        asyncio.run(main_vocal_chatbot_system())
