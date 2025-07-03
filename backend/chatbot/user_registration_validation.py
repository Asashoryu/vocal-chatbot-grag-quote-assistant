import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf  # Already in your code
from contextlib import contextmanager
import nemo.collections.asr as nemo_asr
import torchaudio.transforms as T
from torch.nn.functional import cosine_similarity
import operator  # For sorting the results
import shutil  # For deleting directories

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
SAMPLE_RATE = 16000  # This is the target sample rate for the model
AUDIO_RECORDINGS_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/audio_recordings"
# This is where "registered" embeddings will be saved
EMBEDDING_OUTPUT_DIR = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/audio/speaker_embeddings"
# A common threshold for speaker verification (adjust as needed)
VERIFICATION_THRESHOLD = 0.7

# Ensure directories exist at script start
os.makedirs(AUDIO_RECORDINGS_DIR, exist_ok=True)
os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)

# --- Load Speaker Verification Model (loaded once globally) ---
print("Attempting to load pre-trained Speaker Verification model (TitaNet-Large)...")
verification_model = None
try:
    verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_large")
    print("Successfully loaded pre-trained TitaNet-Large model.")
    # Ensure model is on CPU if CUDA is not available, as per previous logs
    if not torch.cuda.is_available():
        verification_model = verification_model.to('cpu')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an active internet connection and that the model name is correct.")
    print("If you have a local .nemo file, you can try restoring it like this:")
    print("e.g., verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from('path/to/your/titanet-large-finetune.nemo')")

    print("\n--- Listing available Speaker Verification models for debugging ---")
    try:
        available_models = nemo_asr.models.EncDecSpeakerLabelModel.list_available_models()
        if available_models:
            print("Available models:")
            for model_info in available_models:
                print(
                    f"  - {model_info.pretrained_model_name} (Description: {model_info.description})")
            print(
                "\nPlease check the list above for the exact name of the TitaNet-Large model.")
        else:
            print(
                "No available models found. This might indicate an issue with NeMo's connection to NGC.")
    except Exception as list_e:
        print(f"Could not retrieve list of available models: {list_e}")


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

        # print("\n--- Checking Audio Devices with sounddevice ---") # Removed for less verbosity in interactive mode
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]

        if not input_devices:
            print(
                "No input audio devices (microphones) found by sounddevice. Cannot record.")
            return None
        else:
            # print("Available input devices:") # Removed for less verbosity in interactive mode
            # for i, device in enumerate(input_devices):
            #     print(f"  {i}: {device['name']} (Input Channels: {device['max_input_channels']})")

            default_input_device_id = None
            try:
                default_input_device_id = sd.default.device[0]
                # print(f"Attempting to use default sounddevice input device: {devices[default_input_device_id]['name']}") # Removed for less verbosity
            except Exception:
                default_input_device_id = input_devices[0]['index']
                # print(f"Could not determine default sounddevice, using first available: {input_devices[0]['name']}") # Removed for less verbosity

            recording = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate,
                               channels=1, dtype='float32', device=default_input_device_id)
            sd.wait()
            print("Recording finished via sounddevice.")
            sf.write(audio_filepath, recording, sample_rate)
            print(f"Audio saved to: {audio_filepath}")

        return audio_filepath

    except Exception as e:
        print(f"\nAn error occurred during recording: {e}")
        print("Please ensure:")
        print("1. Your microphone is connected and working.")
        print("2. All necessary Python packages are correctly installed (sounddevice, soundfile).")
        return None


# --- Function to Extract Speaker Embedding ---
def extract_speaker_embedding(model, audio_filepath, target_sample_rate=SAMPLE_RATE, embedding_output_dir=EMBEDDING_OUTPUT_DIR, embedding_output_filename="speaker_embedding.pkl", save_embedding=True):
    """
    Extracts speaker embedding from an audio file using the provided NeMo model
    and optionally saves the embedding. Handles resampling to target_sample_rate if needed.
    """
    if model is None:
        print("Error: Speaker verification model is not loaded. Cannot extract embedding.")
        return None
    if not os.path.exists(audio_filepath):
        print(
            f"Error: Audio file not found at {audio_filepath}. Cannot extract embedding.")
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
            # print(f"Resampling audio from {sr} Hz to {target_sample_rate} Hz for model input.") # Removed for less verbosity
            resampler = T.Resample(
                orig_freq=sr, new_freq=target_sample_rate).to(device)
            audio_signal_tensor = resampler(audio_signal_tensor)

        max_amplitude = torch.max(torch.abs(audio_signal_tensor)).item()

        if max_amplitude < 0.001:
            print(
                f"\nWARNING: Audio file '{os.path.basename(audio_filepath)}' appears to be silent or extremely quiet after processing.")
            print(
                "This may lead to poor embedding quality. Please ensure the audio has clear speech.")

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
                    print(
                        f"\nWARNING: L2 norm of the raw extracted embedding for '{os.path.basename(audio_filepath)}' is zero or very close to zero.")
                    print(
                        "This indicates the model produced a silent/empty embedding. It will not be normalized.")

        if save_embedding:
            os.makedirs(embedding_output_dir, exist_ok=True)
            embedding_filepath = os.path.join(
                embedding_output_dir, embedding_output_filename)
            with open(embedding_filepath, 'wb') as f:
                pkl.dump(embedding, f)
            print(f"Embedding saved to: {embedding_filepath}")

        return embedding

    except Exception as e:
        print(
            f"\nAn error occurred during embedding extraction for '{os.path.basename(audio_filepath)}': {e}")
        print("Please ensure:")
        print("1. The audio file is valid and readable.")
        print("2. The NeMo model loaded successfully without errors.")
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
        print(
            "Error: Speaker verification model is not loaded. Cannot perform verification.")
        return False, None, None, []
    if not os.path.exists(new_voice_audio_path):
        print(
            f"Error: New voice audio file not found at {new_voice_audio_path}. Cannot perform verification.")
        return False, None, None, []
    if not os.path.isdir(registered_embeddings_folder):
        print(
            f"Error: Registered embeddings folder not found at {registered_embeddings_folder}.")
        print("Please ensure the folder exists and contains .pkl or .npy embedding files.")
        return False, None, None, []

    print(f"\n--- Starting Speaker Verification Against Registered Embeddings ---")
    print(
        f"Verifying '{os.path.basename(new_voice_audio_path)}' against embeddings in '{registered_embeddings_folder}'...")

    # 1. Extract embedding for the new voice
    print("\nExtracting embedding for the new voice to be verified...")
    new_voice_embedding = extract_speaker_embedding(
        model=model,
        audio_filepath=new_voice_audio_path,
        target_sample_rate=target_sample_rate,
        save_embedding=False  # Don't save this temporary embedding
    )
    if new_voice_embedding is None:
        print("Failed to extract embedding for the new voice. Aborting verification.")
        return False, None, None, []

    new_voice_embedding_tensor = torch.from_numpy(
        new_voice_embedding).unsqueeze(0)

    all_results = []

    # 2. Iterate through registered embeddings
    registered_files = [f for f in os.listdir(
        registered_embeddings_folder) if f.endswith(('.pkl', '.npy'))]

    if not registered_files:
        print(
            f"No .pkl or .npy embedding files found in '{registered_embeddings_folder}'.")
        return False, None, None, []

    print(
        f"\nFound {len(registered_files)} registered embeddings. Comparing...")

    for filename in registered_files:
        filepath = os.path.join(registered_embeddings_folder, filename)
        try:
            with open(filepath, 'rb') as f:
                registered_embedding = pkl.load(f)

            if not isinstance(registered_embedding, np.ndarray) or registered_embedding.ndim != 1:
                print(
                    f"Skipping '{filename}': Invalid embedding format or shape.")
                continue

            registered_embedding_tensor = torch.from_numpy(
                registered_embedding).unsqueeze(0)

            score = cosine_similarity(
                new_voice_embedding_tensor, registered_embedding_tensor).item()
            is_same_speaker = score >= similarity_threshold

            all_results.append(
                {'filename': filename, 'score': score, 'is_match': is_same_speaker})

            print(
                f"  - vs '{filename}': Score = {score:.4f} -> {'MATCH' if is_same_speaker else 'NO MATCH'}")

        except Exception as e:
            print(f"Error processing '{filename}': {e}. Skipping this file.")
            continue

    # 3. Sort results by similarity score (descending)
    all_results_sorted = sorted(
        all_results, key=operator.itemgetter('score'), reverse=True)

    best_overall_score = -1.0
    best_overall_match_filename = None
    best_match_exceeds_threshold = False

    # 4. Find the best match that is also above the threshold
    print("\n--- Detailed Results (Sorted by Similarity) ---")
    if not all_results_sorted:
        print("No valid comparisons were made.")
    else:
        for result in all_results_sorted:
            filename = result['filename']
            score = result['score']
            is_match = result['is_match']
            print(
                f"  - {filename:<30} | Score: {score:.4f} | {'Match Found!' if is_match else 'No Match.'}")

            if is_match and score > best_overall_score:  # Find the best score that also meets threshold
                best_overall_score = score
                best_overall_match_filename = filename
                best_match_exceeds_threshold = True

    print("\n--- Final Verification Summary ---")
    if best_match_exceeds_threshold:
        print(
            f"SUCCESS: The new voice is identified as '{best_overall_match_filename}'!")
        print(
            f"         Similarity Score: {best_overall_score:.4f} (Above threshold: {similarity_threshold:.4f})")
    elif all_results_sorted:
        highest_score_overall = all_results_sorted[0]['score']
        highest_score_filename = all_results_sorted[0]['filename']
        print(f"NO MATCH: The new voice is likely DIFFERENT from all registered speakers.")
        print(
            f"          Highest similarity found was with '{highest_score_filename}' at {highest_score_overall:.4f} (Below threshold: {similarity_threshold:.4f}).")
    else:
        print("No registered embeddings were processed, or an error occurred. Cannot verify speaker.")

    return best_match_exceeds_threshold, best_overall_match_filename, best_overall_score, all_results_sorted


# --- Interactive Main Function ---
def interactive_main():
    if verification_model is None:
        print("\nERROR: Speaker verification model could not be loaded. Exiting interactive mode.")
        return

    while True:
        print("\n" + "="*50)
        print("SPEAKER VERIFICATION SYSTEM MENU")
        print("="*50)
        print("1. Register a new speaker (record audio & save embedding)")
        print("2. Verify a new voice against registered speakers (record audio)")
        print("3. Convert an existing audio file to embedding (save to disk)")
        print("4. List all registered speaker embeddings")
        print("5. Delete a registered speaker embedding")
        print("6. Clear ALL registered speaker embeddings")
        print("7. Exit")
        print("="*50)

        choice = input("Enter your choice (1-7): ").strip()

        if choice == '1':
            print("\n--- Register New Speaker ---")
            user_id = input(
                "Enter a unique ID for the speaker (e.g., 'Alice', 'Bob'): ").strip()
            if not user_id:
                print("Speaker ID cannot be empty. Returning to menu.")
                continue

            output_wav_filename = f"{user_id}_registered_voice.wav"
            recorded_audio_path = record_audio(
                duration_seconds=RECORDING_DURATION_SECONDS,
                output_filename=output_wav_filename,
                prompt_message=f"Recording voice for '{user_id}'. Please speak clearly."
            )

            if recorded_audio_path:
                print(f"Extracting embedding for '{user_id}'...")
                extracted_embedding = extract_speaker_embedding(
                    model=verification_model,
                    audio_filepath=recorded_audio_path,
                    embedding_output_filename=f"{user_id}_embedding.pkl",
                    save_embedding=True
                )
                if extracted_embedding is not None:
                    print(f"Successfully registered '{user_id}'.")
                else:
                    print(f"Failed to register '{user_id}'.")
            else:
                print("Audio recording failed. Cannot register speaker.")

        elif choice == '2':
            print("\n--- Verify New Voice ---")
            if not os.listdir(EMBEDDING_OUTPUT_DIR):
                print(
                    f"No registered embeddings found in '{EMBEDDING_OUTPUT_DIR}'. Please register speakers first.")
                continue

            voice_to_verify_filename = "current_voice_to_verify.wav"
            recorded_audio_path = record_audio(
                duration_seconds=RECORDING_DURATION_SECONDS,
                output_filename=voice_to_verify_filename,
                prompt_message="Recording voice for verification. Please speak clearly."
            )

            if recorded_audio_path:
                best_match_found, best_match_filename, best_match_score, all_results_sorted = \
                    verify_speaker_against_folder_embeddings(
                        model=verification_model,
                        new_voice_audio_path=recorded_audio_path,
                        registered_embeddings_folder=EMBEDDING_OUTPUT_DIR,
                        similarity_threshold=VERIFICATION_THRESHOLD
                    )
            else:
                print("Audio recording failed. Cannot perform verification.")

        elif choice == '3':
            print("\n--- Convert Audio File to Embedding ---")
            audio_file_path = input(
                "Enter the full path to the audio file (e.g., 'my_audio.wav'): ").strip()
            if not os.path.exists(audio_file_path):
                print(f"Error: File not found at '{audio_file_path}'.")
                continue

            output_embedding_name = input(
                "Enter a name for the output embedding (e.g., 'my_custom_voice'): ").strip()
            if not output_embedding_name:
                print("Output embedding name cannot be empty. Returning to menu.")
                continue

            print(f"Extracting embedding from '{audio_file_path}'...")
            extracted_embedding = extract_speaker_embedding(
                model=verification_model,
                audio_filepath=audio_file_path,
                embedding_output_filename=f"{output_embedding_name}_embedding.pkl",
                save_embedding=True
            )
            if extracted_embedding is not None:
                print(
                    f"Successfully created embedding for '{output_embedding_name}'.")
            else:
                print(
                    f"Failed to create embedding for '{output_embedding_name}'.")

        elif choice == '4':
            print("\n--- List Registered Speaker Embeddings ---")
            registered_files = [f for f in os.listdir(
                EMBEDDING_OUTPUT_DIR) if f.endswith(('.pkl', '.npy'))]
            if not registered_files:
                print(
                    f"No registered embeddings found in '{EMBEDDING_OUTPUT_DIR}'.")
            else:
                print(f"Embeddings found in '{EMBEDDING_OUTPUT_DIR}':")
                for i, filename in enumerate(registered_files):
                    print(f"  {i+1}. {filename}")

        elif choice == '5':
            print("\n--- Delete Registered Speaker Embedding ---")
            registered_files = [f for f in os.listdir(
                EMBEDDING_OUTPUT_DIR) if f.endswith(('.pkl', '.npy'))]
            if not registered_files:
                print(
                    f"No registered embeddings found in '{EMBEDDING_OUTPUT_DIR}'.")
                continue

            print(f"Embeddings found in '{EMBEDDING_OUTPUT_DIR}':")
            for i, filename in enumerate(registered_files):
                print(f"  {i+1}. {filename}")

            try:
                file_index_to_delete = int(
                    input("Enter the number of the embedding to delete: ").strip()) - 1
                if 0 <= file_index_to_delete < len(registered_files):
                    filename_to_delete = registered_files[file_index_to_delete]
                    filepath_to_delete = os.path.join(
                        EMBEDDING_OUTPUT_DIR, filename_to_delete)
                    os.remove(filepath_to_delete)
                    print(f"Successfully deleted '{filename_to_delete}'.")
                else:
                    print("Invalid number. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"An error occurred while deleting: {e}")

        elif choice == '6':
            print("\n--- Clear ALL Registered Speaker Embeddings ---")
            confirm = input(
                f"Are you sure you want to delete ALL embeddings in '{EMBEDDING_OUTPUT_DIR}'? (yes/no): ").strip().lower()
            if confirm == 'yes':
                try:
                    shutil.rmtree(EMBEDDING_OUTPUT_DIR)
                    # Recreate empty directory
                    os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)
                    print(
                        f"Successfully cleared all embeddings from '{EMBEDDING_OUTPUT_DIR}'.")
                except Exception as e:
                    print(f"An error occurred while clearing embeddings: {e}")
            else:
                print("Operation cancelled.")

        elif choice == '7':
            print("\nExiting Speaker Verification System. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


# --- Run the interactive main function ---
if __name__ == "__main__":
    interactive_main()
