import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf
from contextlib import contextmanager
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
import torchaudio.transforms as T
from torch.nn.functional import cosine_similarity
import operator
import shutil
import asyncio  # NEW: For async operations
import json  # NEW: For parsing JSON responses, e.g., from LLM API
# NEW: For making actual HTTP requests to Gemini API (though we're moving to local LLM, requests might still be used for local LLM APIs)
import requests

import ollama  # Keep this import at the top of your file

# NEW: Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Define a fallback for torch.cuda.amp.autocast if it's not available
try:
    from torch.cuda.amp import autocast
except ImportError:
    @contextmanager
    def autocast(enabled=None):
        """
        A mock autocast context manager for environments where torch.cuda.amp is not available.
        """
        yield

# --- Global Parameters (can be adjusted) ---
RECORDING_DURATION_SECONDS = 5  # Default duration for new recordings
SAMPLE_RATE = 16000  # This is the target sample rate for ASR and Speaker Recognition models
# NEW: Common sample rate for TTS models (e.g., HiFi-GAN)
TTS_SAMPLE_RATE = 22050
AUDIO_RECORDINGS_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/audio_recordings"
# This is where "registered" embeddings will be saved
EMBEDDING_OUTPUT_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/speaker_embeddings"
# A common threshold for speaker verification (adjust as needed)
VERIFICATION_THRESHOLD = 0.7

# NEW: Flag to control TTS for computer answers
USE_TTS_FOR_ANSWERS = True

# Ensure directories exist at script start
os.makedirs(AUDIO_RECORDINGS_DIR, exist_ok=True)
os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)

# --- Helper function for TTS-enabled printing ---


def print_and_speak(text, tts_model_obj=None, vocoder_model_obj=None, speaker_embedding_obj=None, use_tts_flag=True):
    """Prints text to console and optionally speaks it using TTS."""
    print(text)
    if use_tts_flag and tts_model_obj and vocoder_model_obj:
        print("(Speaking response...)")
        synthesize_speech(tts_model_obj, vocoder_model_obj, text, speaker_embedding_obj,
                          output_filename="last_response.wav", output_dir=AUDIO_RECORDINGS_DIR)


# --- Load Speaker Verification Model (loaded once globally) ---
print("Attempting to load pre-trained Speaker Verification model (TitaNet-Large)...")
verification_model = None
try:
    verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_large")
    print("Successfully loaded pre-trained TitaNet-Large model.")
    if not torch.cuda.is_available():
        verification_model = verification_model.to('cpu')
except Exception as e:
    print(f"Error loading TitaNet model: {e}")
    print("Speaker verification functionality may be limited.")
    print("\n--- Listing available Speaker Verification models for debugging ---")
    try:
        available_models = nemo_asr.models.EncDecSpeakerLabelModel.list_available_models()
        if available_models:
            print("Available models:")
            for model_info in available_models:
                print(
                    f"  - {model_info.pretrained_model_name} (Description: {model_info.description})")
            print(
                "\nPlease check the list above for the exact name of the TitaNet-Large model.")
        else:
            print(
                "No available models found. This might indicate an issue with NeMo's connection to NGC.")
    except Exception as list_e:
        print(f"Could not retrieve list of available models: {list_e}")

# --- NEW: Load ASR Model (Conformer-Large) ---

asr_model = None
print("\nAttempting to load ASR model...")
try:
    # --- CORRECTED LINE HERE ---
    # Use the 'nvidia/' prefix to explicitly fetch from HuggingFace
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/stt_en_fastconformer_transducer_large")

    asr_model.eval()  # Always set to eval mode for inference
    print("Successfully loaded ASR model from HuggingFace.")
except Exception as e:
    print(f"Error loading ASR model: {e}")
    print("ASR functionality may be limited or unavailable.")

# --- NEW: Load TTS Models (FastPitch + HiFi-GAN) ---
print("\nAttempting to load pre-trained TTS models (FastPitch & HiFi-GAN)...")
tts_model = None
vocoder_model = None
try:
    tts_model = nemo_tts.models.FastPitchModel.from_pretrained(
        model_name="tts_en_fastpitch")
    print("Successfully loaded FastPitch TTS model.")
    if not torch.cuda.is_available():
        tts_model = tts_model.to('cpu')

    vocoder_model = nemo_tts.models.HifiGanModel.from_pretrained(
        model_name="tts_en_hifigan")
    print("Successfully loaded HiFi-GAN vocoder model.")
    if not torch.cuda.is_available():
        vocoder_model = vocoder_model.to('cpu')

    print("TTS models loaded successfully.")
except Exception as e:
    print(f"Error loading TTS models: {e}")
    print("Text-to-Speech functionality may be limited.")


# --- Function to Record Audio ---
def record_audio(duration_seconds=RECORDING_DURATION_SECONDS, sample_rate=SAMPLE_RATE, output_filename="recorded_audio.wav", output_dir=AUDIO_RECORDINGS_DIR, prompt_message="Please speak clearly into your microphone now, at a normal to slightly louder volume."):
    """
    Records audio from the microphone using sounddevice, saves it as a WAV file.
    """
    print(f"\n--- Starting Audio Recording ---")
    print(f"Recording your voice for {duration_seconds} seconds...")
    print(prompt_message)

    audio_filepath = None

    try:
        os.makedirs(output_dir, exist_ok=True)
        audio_filepath = os.path.join(output_dir, output_filename)

        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]

        if not input_devices:
            print_and_speak("No input audio devices (microphones) found by sounddevice. Cannot record.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            return None
        else:
            default_input_device_id = None
            try:
                default_input_device_id = sd.default.device[0]
            except Exception:
                default_input_device_id = input_devices[0]['index']

            recording = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate,
                               channels=1, dtype='float32', device=default_input_device_id)
            sd.wait()
            print_and_speak("Recording finished.", tts_model,
                            vocoder_model, None, USE_TTS_FOR_ANSWERS)
            sf.write(audio_filepath, recording, sample_rate)
            print_and_speak(f"Audio saved to: {audio_filepath}",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        return audio_filepath

    except Exception as e:
        print_and_speak(f"\nAn error occurred during recording: {e}",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        print_and_speak("Please ensure: 1. Your microphone is connected and working. 2. All necessary Python packages are correctly installed (sounddevice, soundfile).",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None


# --- Function to Extract Speaker Embedding ---
def extract_speaker_embedding(model, audio_filepath, target_sample_rate=SAMPLE_RATE, embedding_output_dir=EMBEDDING_OUTPUT_DIR, embedding_output_filename="speaker_embedding.pkl", save_embedding=True):
    """
    Extracts speaker embedding from an audio file using the provided NeMo model
    and optionally saves the embedding. Handles resampling to target_sample_rate if needed.
    """
    if model is None:
        print_and_speak("Error: Speaker verification model is not loaded. Cannot extract embedding.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None
    if not os.path.exists(audio_filepath):
        print_and_speak(f"Error: Audio file not found at {audio_filepath}. Cannot extract embedding.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        audio_signal, sr = sf.read(audio_filepath)

        # Convert to a PyTorch tensor
        audio_signal_tensor = torch.from_numpy(audio_signal).float().to(device)

        # --- ROBUST SHAPE HANDLING ---
        if audio_signal_tensor.ndim == 2 and audio_signal_tensor.shape[1] > 1:
            audio_signal_tensor = torch.mean(
                audio_signal_tensor, dim=1, keepdim=False)
        elif audio_signal_tensor.ndim == 2 and audio_signal_tensor.shape[1] == 1:
            audio_signal_tensor = audio_signal_tensor.squeeze(1)

        if audio_signal_tensor.ndim == 1:
            audio_signal_tensor = audio_signal_tensor.unsqueeze(0)
        elif audio_signal_tensor.ndim != 2:
            raise ValueError(
                f"Unexpected audio tensor shape after processing: {audio_signal_tensor.shape}. Expected 2D (batch, time).")

        if sr != target_sample_rate:
            resampler = T.Resample(
                orig_freq=sr, new_freq=target_sample_rate).to(device)
            audio_signal_tensor = resampler(audio_signal_tensor)

        max_amplitude = torch.max(torch.abs(audio_signal_tensor)).item()

        if max_amplitude < 0.001:
            print_and_speak(f"\nWARNING: Audio file '{os.path.basename(audio_filepath)}' appears to be silent or extremely quiet after processing.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            print_and_speak("This may lead to poor embedding quality. Please ensure the audio has clear speech.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        audio_signal_len_tensor = torch.tensor(
            [audio_signal_tensor.shape[1]]).to(device)

        with torch.no_grad():
            with autocast():
                _, embs = model.forward(
                    input_signal=audio_signal_tensor, input_signal_length=audio_signal_len_tensor)
                embedding = embs.cpu().detach().numpy().squeeze(0)

                raw_embs_l2_norm = np.linalg.norm(embedding)

                if raw_embs_l2_norm > 1e-6:
                    embedding = embedding / raw_embs_l2_norm
                else:
                    print_and_speak(f"\nWARNING: L2 norm of the raw extracted embedding for '{os.path.basename(audio_filepath)}' is zero or very close to zero.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                    print_and_speak("This indicates the model produced a silent/empty embedding. It will not be normalized.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        if save_embedding:
            os.makedirs(embedding_output_dir, exist_ok=True)
            embedding_filepath = os.path.join(
                embedding_output_dir, embedding_output_filename)
            with open(embedding_filepath, 'wb') as f:
                pkl.dump(embedding, f)
            print_and_speak(f"Embedding saved to: {embedding_filepath}",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        return embedding

    except Exception as e:
        print_and_speak(f"\nAn error occurred during embedding extraction for '{os.path.basename(audio_filepath)}': {e}",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        print_and_speak("Please ensure: 1. The audio file is valid and readable. 2. The NeMo model loaded successfully without errors.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None


# --- Function for Speaker Verification against a folder of embeddings ---
def verify_speaker_against_folder_embeddings(
    model,
    new_voice_audio_path,
    registered_embeddings_folder=EMBEDDING_OUTPUT_DIR,
    target_sample_rate=SAMPLE_RATE,
    similarity_threshold=VERIFICATION_THRESHOLD
):
    """
    Verifies a new voice against all pre-saved embeddings in a given folder,
    orders the results by similarity, and reports the best match.
    """
    if model is None:
        print_and_speak("Error: Speaker verification model is not loaded. Cannot perform verification.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return False, None, None, []
    if not os.path.exists(new_voice_audio_path):
        print_and_speak(f"Error: New voice audio file not found at {new_voice_audio_path}. Cannot perform verification.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return False, None, None, []
    if not os.path.isdir(registered_embeddings_folder):
        print_and_speak(f"Error: Registered embeddings folder not found at {registered_embeddings_folder}. Please ensure the folder exists and contains .pkl or .npy embedding files.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return False, None, None, []

    print_and_speak(f"\n--- Starting Speaker Verification Against Registered Embeddings ---",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    print_and_speak(f"Verifying '{os.path.basename(new_voice_audio_path)}' against embeddings in '{registered_embeddings_folder}'...",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    # 1. Extract embedding for the new voice
    print_and_speak("\nExtracting embedding for the new voice to be verified...",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    new_voice_embedding = extract_speaker_embedding(
        model=model,
        audio_filepath=new_voice_audio_path,
        target_sample_rate=target_sample_rate,
        save_embedding=False  # Don't save this temporary embedding
    )
    if new_voice_embedding is None:
        print_and_speak("Failed to extract embedding for the new voice. Aborting verification.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return False, None, None, []

    new_voice_embedding_tensor = torch.from_numpy(
        new_voice_embedding).unsqueeze(0)

    all_results = []

    # 2. Iterate through registered embeddings
    registered_files = [f for f in os.listdir(
        registered_embeddings_folder) if f.endswith(('.pkl', '.npy'))]

    if not registered_files:
        print_and_speak(f"No .pkl or .npy embedding files found in '{registered_embeddings_folder}'.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return False, None, None, []

    print_and_speak(f"\nFound {len(registered_files)} registered embeddings. Comparing...",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    for filename in registered_files:
        filepath = os.path.join(registered_embeddings_folder, filename)
        try:
            with open(filepath, 'rb') as f:
                registered_embedding = pkl.load(f)

            if not isinstance(registered_embedding, np.ndarray) or registered_embedding.ndim != 1:
                print_and_speak(f"Skipping '{filename}': Invalid embedding format or shape.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                continue

            registered_embedding_tensor = torch.from_numpy(
                registered_embedding).unsqueeze(0)

            score = cosine_similarity(
                new_voice_embedding_tensor, registered_embedding_tensor).item()
            is_same_speaker = score >= similarity_threshold

            all_results.append(
                {'filename': filename, 'score': score, 'is_match': is_same_speaker})

            # print(f"  - vs '{filename}': Score = {score:.4f} -> {'MATCH' if is_same_speaker else 'NO MATCH'}") # Removed for less verbosity

        except Exception as e:
            print_and_speak(f"Error processing '{filename}': {e}. Skipping this file.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            continue

    # 3. Sort results by similarity score (descending)
    all_results_sorted = sorted(
        all_results, key=operator.itemgetter('score'), reverse=True)

    best_overall_score = -1.0
    best_overall_match_filename = None
    best_match_exceeds_threshold = False

    # 4. Find the best match that is also above the threshold
    print_and_speak("\n--- Detailed Results (Sorted by Similarity) ---",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    if not all_results_sorted:
        print_and_speak("No valid comparisons were made.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    else:
        for result in all_results_sorted:
            filename = result['filename']
            score = result['score']
            is_match = result['is_match']
            print_and_speak(f"  - {filename:<30} | Score: {score:.4f} | {'Match Found!' if is_match else 'No Match.'}",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

            if is_match and score > best_overall_score:  # Find the best score that also meets threshold
                best_overall_score = score
                best_overall_match_filename = filename
                best_match_exceeds_threshold = True

    print_and_speak("\n--- Final Verification Summary ---",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    if best_match_exceeds_threshold:
        print_and_speak(f"SUCCESS: The new voice is identified as '{best_overall_match_filename}'!",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        print_and_speak(f"         Similarity Score: {best_overall_score:.4f} (Above threshold: {similarity_threshold:.4f})",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    elif all_results_sorted:
        highest_score_overall = all_results_sorted[0]['score']
        highest_score_filename = all_results_sorted[0]['filename']
        print_and_speak(f"NO MATCH: The new voice is likely DIFFERENT from all registered speakers.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        print_and_speak(f"          Highest similarity found was with '{highest_score_filename}' at {highest_score_overall:.4f} (Below threshold: {similarity_threshold:.4f}).",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    else:
        print_and_speak("No registered embeddings were processed, or an error occurred. Cannot verify speaker.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    return best_match_exceeds_threshold, best_overall_match_filename, best_overall_score, all_results_sorted


# --- NEW: ASR Module ---
def transcribe_audio(model, audio_filepath, sample_rate=SAMPLE_RATE):
    """
    Transcribes an audio file using the provided ASR model.
    """
    if model is None:
        print_and_speak("Error: ASR model is not loaded. Cannot transcribe audio.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None
    if not os.path.exists(audio_filepath):
        print_and_speak(f"Error: Audio file not found at {audio_filepath}. Cannot transcribe.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None

    try:
        print_and_speak(f"\n--- Starting ASR Transcription for '{os.path.basename(audio_filepath)}' ---",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        # NeMo ASR models expect a list of file paths
        transcriptions = model.transcribe([audio_filepath])
        if transcriptions and len(transcriptions) > 0:
            transcribed_text = transcriptions[0]
            print_and_speak(f"Transcription: \"{transcribed_text}\"",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            return transcribed_text
        else:
            print_and_speak("ASR returned no transcription.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            return None
    except Exception as e:
        print_and_speak(f"An error occurred during ASR transcription: {e}",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        return None


# --- Neo4j Connection and Search Functions ---
# IMPORTANT: Replace these with your actual Neo4j connection details.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "amonolexandr"  # <--- REPLACE WITH YOUR ACTUAL NEO4J PASSWORD

# Mapping of user-friendly entity types to their corresponding Full-Text Index names
ENTITY_INDEX_MAP = {
    "1": {"name": "Author", "index": "authorNamesIndex"},
    "2": {"name": "Quote", "index": "quoteTextsIndex"},
    "3": {"name": "Context", "index": "contextTextsIndex"},
    "4": {"name": "Detail", "index": "detailTextsIndex"},
    # For searching across all indexed text
    "5": {"name": "All Content", "index": "allTextContentIndex"}
}


class Neo4jConnector:
    """
    Manages the connection to the Neo4j database.
    Encapsulates driver initialization and closing.
    """

    def __init__(self, uri, username, password):
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None

    def connect(self):
        """Initializes the Neo4j driver."""
        try:
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._username, self._password))
            # Verify connectivity by attempting a simple query
            self._driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
            return True
        except ServiceUnavailable as e:
            print(
                f"Connection failed: Neo4j database is not running or accessible at {self._uri}. Error: {e}")
            return False
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Please check your URI, username, and password.")
            return False

    def close(self):
        """Closes the Neo4j driver."""
        if self._driver:
            self._driver.close()
            print("Neo4j driver closed.")

    def get_driver(self):
        """Returns the initialized Neo4j driver."""
        return self._driver


def search_entity(driver, entity_choice: str, substring: str, k: int = 10):
    """
    Performs a full-text search on the selected entity type.

    Args:
        driver (neo4j.GraphDatabase.driver): The Neo4j database driver.
        entity_choice (str): The user's choice (e.g., "1" for Author).
        substring (str): The substring to search for.
        k (int): The maximum number of top results to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a search result.
              Returns an empty list if no results or an error occurs.
    """
    if not driver:
        print("Error: Neo4j driver is not initialized.")
        return []

    entity_info = ENTITY_INDEX_MAP.get(entity_choice)
    if not entity_info:
        print("Invalid entity choice.")
        return []

    index_name = entity_info["index"]
    entity_name = entity_info["name"]
    results = []

    # Use a session to interact with the database
    with driver.session() as session:
        # Construct the search term with a wildcard for prefix matching
        search_term = f"{substring}*"

        # Cypher query for full-text search
        # The CASE statement dynamically selects the correct property based on node label
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
            print(
                f"\nSearching for '{substring}' in '{entity_name}' (top {k} results)...")
            result = session.run(query, searchTerm=search_term, k=k)
            for record in result:
                results.append({
                    # Join labels for display
                    "NodeType": ", ".join(record["NodeType"]),
                    "Content": record["Content"],
                    "Score": record["score"]
                })
        except Exception as e:
            print(f"An error occurred during the search: {e}")
            print(
                f"Please ensure the full-text index '{index_name}' exists and is online.")
    return results

# --- Placeholder for a local open-source LLM interaction ---
# This is a critical part you'll need to implement based on your chosen local LLM setup.
# Examples include using the 'ollama' library for Ollama, 'transformers' library for local models,
# or making an HTTP request to a local LLM API (e.g., vLLM, text-generation-webui).


async def call_local_llm(prompt: str) -> str:
    """
    Interacts with a local Ollama LLM to generate a response based on the prompt.

    Args:
        prompt (str): The prompt to send to the LLM, typically including user query and RAG context.

    Returns:
        str: The generated response from the Ollama model.
    """
    print_and_speak("--- Calling local LLM (via Ollama) ---", tts_model,
                    vocoder_model, None, USE_TTS_FOR_ANSWERS)
    # Print first 500 chars of prompt for debugging
    print(f"LLM Prompt:\n{prompt[:500]}...")

    # Define the Ollama model you want to use.
    # Make sure this model is pulled locally via 'ollama pull <model_name>'
    # Example lightweight models: 'llama2', 'mistral', 'gemma:2b', 'phi3'
    # For a very light and performant model, 'phi3' or 'gemma:2b' are good choices.
    ollama_model_name = "tinyllama"  # Or "gemma:2b", "llama2", "mistral", etc.

    try:
        # Use ollama.chat for a more conversational style, which is suitable for RAG scenarios.
        # This sends a list of messages. For a single turn, it's just the user message.
        response = ollama.chat(
            model=ollama_model_name,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            stream=False  # Set to True if you want to process the response token by token
        )

        # Extract the content from the response
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return f"Error: Ollama response structure was unexpected for model '{ollama_model_name}'. No 'message' or 'content' key found."

    except ollama.ResponseError as e:
        # This catches errors specific to Ollama's API responses
        return f"Error interacting with Ollama: {e}. Please ensure the Ollama server is running and the model '{ollama_model_name}' is pulled locally."
    except Exception as e:
        # General error catching for other issues (e.g., network, unexpected library errors)
        return f"An unexpected error occurred while calling the local LLM: {e}. Check Ollama server status or network."

# --- NEW: Chatbot Module (Graph RAG with Local LLM) ---


async def query_neo4j_rag(user_query: str, recognized_user_id: str = "unknown") -> str:
    """
    Function for querying the Neo4j graph and performing RAG with a local LLM.
    """
    print_and_speak(f"\n--- Chatbot: Processing Query for '{recognized_user_id}' ---",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
    print_and_speak(f"User Query (ASR): \"{user_query}\"",
                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    neo4j_connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    if not neo4j_connector.connect():
        return "I'm sorry, I cannot connect to the knowledge base at this moment. Please try again later."

    retrieved_data = []
    try:
        # --- Neo4j Graph Retrieval ---
        # Attempt to identify the type of query to guide the search in Neo4j.
        # This uses simple keyword matching; for more robust applications,
        # consider using an NLP model to extract entities/intents from `user_query`.

        search_term = user_query
        entity_choice = "5"  # Default to searching all content index

        # Simple keyword-based intent recognition for better retrieval targeting
        if "author of" in user_query.lower() or "who said" in user_query.lower() or "tell me about" in user_query.lower().split():
            entity_choice = "1"  # Author
        elif "quote about" in user_query.lower() or "what is the quote" in user_query.lower():
            entity_choice = "2"  # Quote
        elif "context of" in user_query.lower():
            entity_choice = "3"  # Context
        elif "detail of" in user_query.lower() or "source of" in user_query.lower():
            entity_choice = "4"  # Detail
        # If the user asks directly for a quote and it's long, use it as search term
        elif ("quote:" in user_query.lower() or "said:" in user_query.lower()) and len(user_query) > 10:
            entity_choice = "2"  # Likely looking for a specific quote
            # Extract text after "quote:" or "said:"
            search_term = user_query.split(':', 1)[-1].strip()

        # Perform the search in Neo4j
        search_results = search_entity(
            neo4j_connector.get_driver(), entity_choice, search_term, k=5)  # Limit to top 5 relevant results

        if search_results:
            for i, res in enumerate(search_results):
                retrieved_data.append(
                    f"Result {i+1}: Type: {res['NodeType']}, Content: \"{res['Content']}\" (Score: {res['Score']:.2f})")
            context_string_for_llm = "\n".join(retrieved_data)
        else:
            context_string_for_llm = "No specific content found in the graph relevant to the query."

    except Exception as e:
        print_and_speak(
            f"Error during Neo4j retrieval: {e}", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
        context_string_for_llm = "An error occurred while searching the knowledge base. Please try rephrasing your question."
    finally:
        neo4j_connector.close()

    # --- LLM for Response Generation (using local open-source LLM) ---
    # print_and_speak("Generating natural language response using local LLM...",
    #                 tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    llm_prompt = f"""You are a helpful chatbot that answers questions based on provided context from a graph database of quotes.
    User's query: "{user_query}"
    Context from Neo4j graph:
    {context_string_for_llm}

    Based on the user's query and the provided context, generate a concise, accurate, and helpful response.
    If the context explicitly states "No specific content found" just say that no match was found.
    Prioritize using the retrieved information directly when it's relevant to the user's question about quotes, authors, or details.
    If multiple results are provided in the context, integrate them coherently into your answer. IMPORTANT: BE VERY CONCISE OR I DELETE YOU!!!
    """

    llm_response = await call_local_llm(llm_prompt)

    return llm_response

# --- NEW: Personalized Text-to-Speech (TTS) Module ---


def synthesize_speech(tts_model_obj, vocoder_model_obj, text: str, speaker_embedding: np.ndarray = None, output_filename="response.wav", output_dir=AUDIO_RECORDINGS_DIR):
    """
    Synthesizes speech from text using NeMo TTS models.
    Can optionally be conditioned by a speaker embedding for personalization.
    """
    if tts_model_obj is None or vocoder_model_obj is None:
        print("Error: TTS models are not loaded. Cannot synthesize speech.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

        # Prepare input text
        parsed_text = tts_model_obj.parse(text)

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        speaker_embedding_tensor = None
        if speaker_embedding is not None:
            # Ensure embedding is 2D (batch_size, embedding_dim)
            if speaker_embedding.ndim == 1:
                speaker_embedding_tensor = torch.from_numpy(
                    speaker_embedding).unsqueeze(0).to(device)
            elif speaker_embedding.ndim == 2:
                speaker_embedding_tensor = torch.from_numpy(
                    speaker_embedding).to(device)
            else:
                print(
                    "Warning: Speaker embedding has an unexpected shape. Not using for TTS.")
                speaker_embedding_tensor = None

        # Prepare arguments for generate_spectrogram
        spectrogram_args = {
            "tokens": parsed_text.to(device)
        }

        # ONLY pass speaker_emb if a valid tensor is available
        if speaker_embedding_tensor is not None:
            spectrogram_args["speaker_emb"] = speaker_embedding_tensor

        # Generate spectrogram
        with torch.no_grad():
            with autocast():  # This is likely where float16 is introduced
                # Unpack the dictionary to pass arguments conditionally
                spectrogram = tts_model_obj.generate_spectrogram(
                    **spectrogram_args)

        # Generate audio from spectrogram using vocoder
        with torch.no_grad():
            with autocast():  # This too
                audio = vocoder_model_obj(spec=spectrogram)

        audio = audio.cpu().numpy().squeeze()

        # --- IMPORTANT FIX HERE: Convert to float32 before saving/playing ---
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
        # --- END FIX ---

        # Normalize audio to avoid clipping, if necessary
        if np.abs(audio).max() > 0:
            # Normalize to 90% of max amplitude
            audio = audio / np.abs(audio).max() * 0.9

        # Save and play audio
        sf.write(output_filepath, audio, TTS_SAMPLE_RATE)
        sd.play(audio, TTS_SAMPLE_RATE)
        sd.wait()

        return output_filepath

    except Exception as e:
        print(f"An error occurred during TTS synthesis: {e}")
        return None

# --- Interactive Main Function ---


async def interactive_main():  # Changed to async to support await in query_neo4j_rag
    global USE_TTS_FOR_ANSWERS  # Allow modifying the global flag within this function

    if verification_model is None or asr_model is None or tts_model is None or vocoder_model is None:
        print_and_speak("\nERROR: One or more required models could not be loaded. Some functionalities may be unavailable.",
                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

    # Initial check and prompt for TTS
    if USE_TTS_FOR_ANSWERS:
        print_and_speak("Text to speech for computer answers is currently **enabled**.",
                        tts_model, vocoder_model, None, False)  # Don't speak this message twice
        print("Text to speech for computer answers is currently **enabled**.")
    else:
        print("Text to speech for computer answers is currently **disabled**.")

    while True:
        print("\n" + "="*50)
        print("VOCAL CHATBOT SYSTEM MENU")
        print("="*50)
        print("1. Register a new speaker")
        print("2. Authenticate User (Voice Identification)")  # New option
        print("3. Start Voice Chat (Authenticated)")  # New option
        print("4. Query Neo4j (Text Input)")  # New option
        print("5. Transcribe an audio file (ASR only)")  # Shifted
        print("6. Convert an existing audio file to embedding")  # Shifted
        print("7. Synthesize speech from text (TTS only)")  # Shifted
        print("8. List all registered speaker embeddings")  # Shifted
        print("9. Delete a registered speaker embedding")  # Shifted
        print("10. Clear ALL registered speaker embeddings")  # Shifted
        print(
            f"11. Toggle TTS for computer answers (Currently: {'ENABLED' if USE_TTS_FOR_ANSWERS else 'DISABLED'})")  # Shifted
        print("0. Exit")
        print("="*50)

        choice = input("Enter your choice: ").strip()

        # Moved the original '1. Register a new speaker'
        if choice == '1':
            print_and_speak("\n--- Register New Speaker ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            user_id = input(
                "Enter a unique ID for the speaker (e.g., 'Alice', 'Bob'): ").strip()
            if not user_id:
                print_and_speak("Speaker ID cannot be empty. Returning to menu.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                continue

            output_wav_filename = f"{user_id}_registered_voice.wav"
            recorded_audio_filepath = record_audio(
                output_filename=output_wav_filename,
                prompt_message=f"Please speak for {RECORDING_DURATION_SECONDS} seconds to register '{user_id}'."
            )
            if recorded_audio_filepath:
                embedding_filename = f"{user_id}_embedding.pkl"
                speaker_embedding = extract_speaker_embedding(
                    verification_model,
                    recorded_audio_filepath,
                    embedding_output_filename=embedding_filename,
                    save_embedding=True
                )
                if speaker_embedding is not None:
                    print_and_speak(
                        f"Speaker '{user_id}' registered successfully!", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                else:
                    print_and_speak(
                        f"Failed to register speaker '{user_id}'.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(
                    "Audio recording failed. Speaker not registered.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # New option: Authenticate User
        elif choice == '2':
            print_and_speak("\n--- Authenticate User (Voice Identification) ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            temp_audio_filepath = record_audio(
                output_filename="voice_to_verify.wav",
                prompt_message=f"Please speak for {RECORDING_DURATION_SECONDS} seconds for authentication."
            )

            if temp_audio_filepath:
                is_identified, identified_user_filename, score, all_speaker_results = verify_speaker_against_folder_embeddings(
                    verification_model, temp_audio_filepath)

                recognized_user_id = "unknown"
                if is_identified:
                    recognized_user_id = identified_user_filename.replace(
                        "_embedding.pkl", "")
                    print_and_speak(f"Authentication successful! Identified User: {recognized_user_id}. Score: {score:.4f}",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                else:
                    print_and_speak(f"Authentication failed. User not identified. Best match (if any) was {identified_user_filename.replace('_embedding.pkl', '') if identified_user_filename else 'none'} with score: {score:.4f}",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                    # For more detail in logs, print all speaker results
                    for result_id, result_score in all_speaker_results.items():
                        print(f" - {result_id} | Score: {result_score:.4f}")

                # Clean up the temporary audio file
                if os.path.exists(temp_audio_filepath):
                    os.remove(temp_audio_filepath)
                    print(f"Cleaned up temporary audio: {temp_audio_filepath}")
            else:
                print_and_speak("No audio recorded for authentication.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # New option: Start Voice Chat (Authenticated)
        elif choice == '3':
            print_and_speak("\n--- Starting Voice Chat (Requires prior authentication) ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

            # First, check if there's a recently identified user or prompt for one
            # This part needs state management, for simplicity, we'll re-authenticate or ask.
            print_and_speak("Please authenticate yourself first.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            temp_auth_audio_filepath = record_audio(
                output_filename="chat_auth_voice.wav",
                prompt_message=f"Speak for {RECORDING_DURATION_SECONDS} seconds to authenticate for chat."
            )

            recognized_user_id = "unknown"
            personalization_embedding = None

            if temp_auth_audio_filepath:
                is_identified, identified_user_filename, score, _ = verify_speaker_against_folder_embeddings(
                    verification_model, temp_auth_audio_filepath)

                if is_identified:
                    recognized_user_id = identified_user_filename.replace(
                        "_embedding.pkl", "")
                    print_and_speak(f"Authenticated as: {recognized_user_id}. Now, please state your query.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                    # Load personalization embedding
                    embedding_path = os.path.join(
                        EMBEDDING_OUTPUT_DIR, identified_user_filename)
                    if os.path.exists(embedding_path):
                        with open(embedding_path, 'rb') as f:
                            personalization_embedding = pkl.load(f)
                    else:
                        print(
                            f"Warning: Personalization embedding file not found for {recognized_user_id}. Using default voice.")

                    # Clean up auth audio
                    os.remove(temp_auth_audio_filepath)
                    print(
                        f"Cleaned up temporary auth audio: {temp_auth_audio_filepath}")

                else:
                    print_and_speak("Authentication failed. Cannot start chat.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                    if os.path.exists(temp_auth_audio_filepath):
                        os.remove(temp_auth_audio_filepath)
                    continue  # Go back to main menu

            else:
                print_and_speak("Authentication audio not recorded. Cannot start chat.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                continue  # Go back to main menu

            # Proceed with chat query only if authentication was successful
            temp_query_audio_filepath = record_audio(
                output_filename="current_chat_query.wav",
                prompt_message=f"Please state your query for {RECORDING_DURATION_SECONDS} seconds."
            )

            if temp_query_audio_filepath:
                user_query = transcribe_audio(
                    asr_model, temp_query_audio_filepath)

                if user_query:
                    chatbot_response = await query_neo4j_rag(user_query, recognized_user_id)
                    print_and_speak(f"\n--- Chatbot: {chatbot_response}",
                                    tts_model, vocoder_model, personalization_embedding, USE_TTS_FOR_ANSWERS)
                else:
                    print_and_speak("Could not transcribe audio. Please try speaking more clearly.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

                # Clean up query audio
                if os.path.exists(temp_query_audio_filepath):
                    os.remove(temp_query_audio_filepath)
                    print(
                        f"Cleaned up temporary query audio: {temp_query_audio_filepath}")
            else:
                print_and_speak("No query audio recorded for chat session.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # New option: Query Neo4j (Text Input)
        elif choice == '4':
            print_and_speak("\n--- Query Neo4j (Text Input) ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            user_query = input("Enter your text query: ").strip()
            if not user_query:
                print_and_speak("Query cannot be empty. Returning to menu.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                continue

            # For text queries, we don't have a voice to identify, so user_id is generic
            # You might want to ask for a user ID here if it's relevant for text queries
            recognized_user_id = "text_user"

            chatbot_response = await query_neo4j_rag(user_query, recognized_user_id)
            print_and_speak(f"\n--- Chatbot: {chatbot_response}",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)  # No personalization for text input

        # Original option 3, now 5
        elif choice == '5':
            print_and_speak("\n--- Transcribe Audio File (ASR Only) ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            input_audio_path = input(
                "Enter path to audio file for transcription: ").strip()
            if os.path.exists(input_audio_path):
                # Ensure transcribe_audio function returns the text
                transcribed_text = transcribe_audio(
                    asr_model, input_audio_path)
                if transcribed_text:
                    print(f"Transcription: {transcribed_text}")
                    print_and_speak(f"Transcription complete: {transcribed_text}",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                else:
                    print_and_speak("Failed to transcribe audio.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(f"Error: File not found at {input_audio_path}.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 4, now 6
        elif choice == '6':
            print_and_speak("\n--- Convert Audio to Embedding ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            input_audio_path = input(
                "Enter path to audio file to convert to embedding: ").strip()
            if os.path.exists(input_audio_path):
                output_filename = input(
                    "Enter desired output embedding filename (e.g., 'new_voice_embedding.pkl'): ").strip()
                if not output_filename:
                    output_filename = "manual_embedding.pkl"
                if not output_filename.endswith(('.pkl', '.npy')):
                    output_filename += ".pkl"

                # Check if an embedding was successfully extracted (function should return it)
                extracted_emb = extract_speaker_embedding(
                    verification_model, input_audio_path, embedding_output_filename=output_filename, save_embedding=True)

                if extracted_emb is not None:
                    print_and_speak(f"Embedding successfully created and saved to {os.path.join(EMBEDDING_OUTPUT_DIR, output_filename)}.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                else:
                    print_and_speak(f"Failed to create embedding from {input_audio_path}.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(f"Error: File not found at {input_audio_path}.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 5, now 7
        elif choice == '7':
            print_and_speak("\n--- Synthesize Speech from Text (TTS Only) ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            text_to_synthesize = input("Enter text to synthesize: ").strip()
            if not text_to_synthesize:
                print_and_speak("Text cannot be empty. Returning to menu.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                continue

            use_personalization = input(
                "Use a specific speaker embedding for personalization? (y/n): ").strip().lower()
            speaker_emb_for_tts = None
            if use_personalization == 'y':
                embedding_id = input(
                    "Enter the ID of the registered speaker (e.g., 'Alice'): ").strip()
                embedding_filepath = os.path.join(
                    EMBEDDING_OUTPUT_DIR, f"{embedding_id}_embedding.pkl")
                if os.path.exists(embedding_filepath):
                    try:
                        with open(embedding_filepath, 'rb') as f:
                            speaker_emb_for_tts = pkl.load(f)
                        print_and_speak(f"Using '{embedding_id}' for TTS personalization.",
                                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                    except Exception as e:
                        print_and_speak(f"Error loading embedding for '{embedding_id}': {e}. Using default voice.",
                                        tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                        speaker_emb_for_tts = None  # Ensure it's reset if load fails
                else:
                    print_and_speak(f"Embedding for '{embedding_id}' not found. Using default voice.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

            output_filename_tts = input(
                "Enter desired output audio filename (e.g., 'my_tts_output.wav'): ").strip()
            if not output_filename_tts:
                output_filename_tts = "synthesized_output.wav"
            if not output_filename_tts.endswith('.wav'):
                output_filename_tts += ".wav"

            synthesize_speech(tts_model, vocoder_model, text_to_synthesize,
                              speaker_emb_for_tts, output_filename=output_filename_tts)
            print_and_speak(f"Speech synthesized to {os.path.join(AUDIO_RECORDINGS_DIR, output_filename_tts)}",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 6, now 8
        elif choice == '8':
            print_and_speak("\n--- Registered Speaker Embeddings ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            registered_files = [f for f in os.listdir(
                EMBEDDING_OUTPUT_DIR) if f.endswith(('.pkl', '.npy'))]
            if registered_files:
                for i, filename in enumerate(registered_files):
                    user_id = filename.replace("_embedding.pkl", "")
                    # Filter out any non-embedding files if any
                    if "_embedding.pkl" in filename or "_embedding.npy" in filename:
                        print(f" {i+1}. {user_id} ({filename})")

                # Speak out a summary for TTS
                if registered_files:
                    speakable_list = ", ".join([f.replace("_embedding.pkl", "").replace(
                        "_embedding.npy", "") for f in registered_files])
                    print_and_speak(f"Currently registered speakers are: {speakable_list}.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(
                    "No speaker embeddings registered yet.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 7, now 9
        elif choice == '9':
            print_and_speak("\n--- Delete Registered Speaker Embedding ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            user_id_to_delete = input(
                "Enter the ID of the speaker to delete (e.g., 'Alice'): ").strip()
            embedding_filepath = os.path.join(
                EMBEDDING_OUTPUT_DIR, f"{user_id_to_delete}_embedding.pkl")
            # Also check for .npy if you plan to save them that way
            embedding_filepath_npy = os.path.join(
                EMBEDDING_OUTPUT_DIR, f"{user_id_to_delete}_embedding.npy")

            found_file = False
            if os.path.exists(embedding_filepath):
                os.remove(embedding_filepath)
                found_file = True
            elif os.path.exists(embedding_filepath_npy):
                os.remove(embedding_filepath_npy)
                found_file = True

            if found_file:
                print_and_speak(f"Successfully deleted embedding for '{user_id_to_delete}'.",
                                tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(
                    f"Embedding for '{user_id_to_delete}' not found.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 8, now 10
        elif choice == '10':
            print_and_speak("\n--- Clear ALL Registered Speaker Embeddings ---",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            confirm = input(
                "Are you sure you want to delete ALL registered speaker embeddings? This cannot be undone! (yes/no): ").strip().lower()
            if confirm == 'yes':
                try:
                    # Remove the directory and recreate it to ensure it's empty
                    if os.path.exists(EMBEDDING_OUTPUT_DIR):
                        shutil.rmtree(EMBEDDING_OUTPUT_DIR)
                    os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)
                    print_and_speak("All registered speaker embeddings have been cleared.",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
                except Exception as e:
                    print_and_speak(f"Error clearing embeddings: {e}",
                                    tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            else:
                print_and_speak(
                    "Operation cancelled. No embeddings were deleted.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 9, now 11
        elif choice == '11':
            USE_TTS_FOR_ANSWERS = not USE_TTS_FOR_ANSWERS
            status = "ENABLED" if USE_TTS_FOR_ANSWERS else "DISABLED"
            print_and_speak(f"Text to Speech for computer answers is now {status}.",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

        # Original option 0, remains 0
        elif choice == '0':
            print_and_speak("\nExiting Vocal Chatbot System. Goodbye!",
                            tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)
            break

        else:
            print_and_speak(
                "Invalid choice. Please enter a number from the menu.", tts_model, vocoder_model, None, USE_TTS_FOR_ANSWERS)

if __name__ == "__main__":
    asyncio.run(interactive_main())
