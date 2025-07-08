import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F  # For cosine_similarity
from torch.cuda.amp import autocast
from typing import Optional, Dict, Union, Tuple
import traceback  # Import traceback for detailed error logging
import pickle  # For loading embeddings in verify_speaker
import torchaudio  # For resampling audio in generate_speaker_embedding
import inspect  # This line must be present!

# Assuming config is correctly defined and accessible in your project
import config


class AudioManager:
    """
    Manages all audio-related operations as a singleton.
    It needs to be initialized once with models and configuration.
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of AudioManager is ever created.
        """
        if cls._instance is None:
            cls._instance = super(AudioManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes AudioManager attributes. This runs only once for the singleton.
        """
        if not hasattr(self, '_initialized_flag'):  # Use a distinct flag for __init__
            self.asr_model = None
            self.tts_model = None
            self.vocoder_model = None
            self.speaker_verification_model = None
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Default device

            # These will be set by the initialize method from config
            self.sample_rate: Optional[int] = None
            self.recording_duration_seconds: Optional[int] = None
            self.audio_records_dir: Optional[str] = None
            self.use_tts_for_answers_flag: Optional[bool] = None

            self.num_tts_speakers: int = 0  # Stores the number of pre-trained TTS speakers

            # This stores the embedding from the *last successful call to generate_speaker_embedding*.
            # It's NOT a registry of all users, but a temporary holder for a single embedding.
            self._last_extracted_user_embedding: Optional[np.ndarray] = None

            self._initialized_flag = True  # Mark __init__ as completed

    def initialize(self, asr_model, tts_model, vocoder_model,
                   sample_rate: int, recording_duration_seconds: int,
                   audio_records_dir: str, use_tts_for_answers_flag: bool,
                   speaker_verification_model=None):
        """
        Initializes the AudioManager's internal state with models and configuration.
        This method should be called only ONCE at application startup.
        """
        if self._is_initialized:
            print(
                "AudioManager is already fully initialized. Skipping re-initialization.")
            return

        print("Initializing AudioManager...")
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.vocoder_model = vocoder_model
        self.speaker_verification_model = speaker_verification_model
        self.sample_rate = sample_rate
        self.recording_duration_seconds = recording_duration_seconds
        self.audio_records_dir = audio_records_dir
        self.use_tts_for_answers_flag = use_tts_for_answers_flag
        self.global_speaker_id = 17

        # Device is already set in __init__, but models are moved here
        if self.device == 'cpu':
            print(
                "Warning: Running on CPU. For faster performance, consider using a CUDA-enabled GPU.")

        os.makedirs(self.audio_records_dir, exist_ok=True)

        if self.asr_model:
            self.asr_model.to(self.device)
            self.asr_model.eval()
            print("SUCCESS: ASR model successfully assigned and moved to device.")

        if self.tts_model:
            self.tts_model.to(self.device)
            self.tts_model.eval()
            print("SUCCESS: TTS model successfully assigned and moved to device.")

        if self.vocoder_model:
            self.vocoder_model.to(self.device)
            self.vocoder_model.eval()
            print("SUCCESS: Vocodecoder model successfully assigned and moved to device.")

        if self.speaker_verification_model:
            self.speaker_verification_model.to(self.device)
            self.speaker_verification_model.eval()
            print(
                "SUCCESS: speaker verification model successfully assigned and moved to device.")
        else:
            print("Warning: Speaker verification model not provided during AudioManager initialization. Speaker embedding extraction will be unavailable.")

        self._is_initialized = True
        print("AudioManager initialized successfully.")

    def _check_initialized(self):
        """Internal helper to ensure the manager has been initialized."""
        if not self._is_initialized:
            raise RuntimeError(
                "AudioManager not initialized. Call initialize() first with all required models and configurations.")

    def print_and_speak(self, text: str, speaker_id: int = 0):
        """
        Prints text to console and optionally speaks it using TTS.
        It uses `speaker_id` to determine the speaker for multi-speaker models.
        """
        if speaker_id != 0:
            speaker_id_used = speaker_id
        else:
            speaker_id_used = self.global_speaker_id
        self._check_initialized()
        print(text)
        if self.use_tts_for_answers_flag and self.tts_model and self.vocoder_model:
            print("(Speaking response...)")
            self.synthesize_speech(
                text=text,
                speaker_id=speaker_id_used,
                output_filename="last_response.wav",
                output_dir=self.audio_records_dir,
            )
        elif not self.use_tts_for_answers_flag:
            print("(TTS disabled by configuration.)")
        else:
            print("(TTS models not loaded, cannot speak.)")

    def record_audio(self, prompt_message: str) -> Optional[np.ndarray]:
        """
        Records audio from the microphone for a user-specified duration.
        Returns the recorded audio data as a NumPy array or None on failure.
        """
        self._check_initialized()
        print(f"\n--- Starting Audio Recording Setup ---")
        print(prompt_message)

        recorded_audio_data = None
        try:
            # Query and list input devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                print(
                    "No input audio devices (microphones) found by sounddevice. Cannot record.")
                print("Available sounddevice devices:", devices)
                return None

            # Attempt to use default input device, fall back to first available
            default_input_device_id = sd.default.device[0]
            if default_input_device_id is None and input_devices:
                default_input_device_id = input_devices[0]['index']
            elif default_input_device_id is None:
                print("No default input device and no other input devices found.")
                return None

            print(
                f"Using input device: {sd.query_devices(default_input_device_id)['name']} (ID: {default_input_device_id})")

            # Get recording duration from user
            try:
                duration_input = input(
                    "Enter recording duration in seconds (e.g., 7): ").strip()
                recording_duration_seconds = float(duration_input)
                if recording_duration_seconds <= 0:
                    print("Duration must be a positive number.")
                    return None
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                return None

            print(
                f"\nRecording started for {recording_duration_seconds} seconds...")

            # Record audio using the configured sample rate
            recorded_audio_data = sd.rec(int(recording_duration_seconds * self.sample_rate),
                                         samplerate=self.sample_rate, channels=1, dtype='float32',
                                         device=default_input_device_id)
            sd.wait()  # Wait for recording to finish
            print("Recording finished.")
            return recorded_audio_data

        except Exception as e:
            print(f"An error occurred during recording: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return None

    def transcribe_audio(self, audio_filepath: str) -> str:
        """
        Transcribes an audio file using the ASR model.
        Returns the transcribed text as a string. Returns an empty string if transcription fails.
        """
        self._check_initialized()
        if self.asr_model is None:
            print("Error: ASR model not loaded. Cannot transcribe audio.")
            return ""
        if not os.path.exists(audio_filepath):
            print(
                f"Error: Audio file not found at {audio_filepath}. Cannot transcribe.")
            return ""

        try:
            print(
                f"\n--- Starting ASR Transcription for '{os.path.basename(audio_filepath)}' ---")

            # Perform transcription
            # Assuming ASR model's transcribe method takes a list of file paths
            transcriptions_raw = self.asr_model.transcribe([audio_filepath])
            print(
                f"DEBUG: Raw transcription from ASR model (type: {type(transcriptions_raw)}): {transcriptions_raw}")

            transcribed_text = ""
            processed_transcriptions = []

            # Handle various potential output formats from ASR model
            if isinstance(transcriptions_raw, tuple) and len(transcriptions_raw) > 0:
                if isinstance(transcriptions_raw[0], list):
                    processed_transcriptions = transcriptions_raw[0]
                else:
                    processed_transcriptions = [transcriptions_raw[0]]
                print(
                    f"DEBUG: Handled tuple output, taking first element: {processed_transcriptions}")
            elif isinstance(transcriptions_raw, list):
                processed_transcriptions = transcriptions_raw
                print(
                    f"DEBUG: Handling standard list output: {processed_transcriptions}")
            else:
                print(
                    f"ASR returned unexpected type: {type(transcriptions_raw)}. Expected list or tuple.")
                return ""

            # Flatten any nested lists and join into a single string
            flat_transcriptions = []
            for item in processed_transcriptions:
                if isinstance(item, list):
                    flat_transcriptions.extend(item)
                else:
                    flat_transcriptions.append(item)

            print(
                f"DEBUG: Flattened transcription list: {flat_transcriptions}")

            transcribed_text = " ".join(flat_transcriptions).strip()

            if not transcribed_text:
                print("ASR returned no meaningful transcription.")
                return ""

            print(
                f"DEBUG: Final transcribed text before return: \"{transcribed_text}\"")

            return transcribed_text
        except Exception as e:
            print(f"An error occurred during ASR transcription: {e}")
            traceback.print_exc()
            return ""

    def generate_speaker_embedding(self, audio_filepath: str):
        """
        Generates a speaker embedding from an audio file using the
        speaker_verification_model. The model is expected to handle internal
        loading and preprocessing (including resampling if necessary).

        Args:
            audio_filepath (str): Path to the audio file.

        Returns:
            numpy.ndarray: The speaker embedding as a NumPy array, or None if an error occurs.
        """
        if not self.speaker_verification_model:
            print(
                "ERROR: Speaker verification model not loaded. Cannot generate embeddings.")
            return None

        try:
            # CRITICAL CORRECTION: NeMo's get_embedding for EncDecSpeakerLabelModel
            # typically expects a list of file paths. It handles the internal
            # audio loading, resampling, and feature extraction.
            embeddings_list = self.speaker_verification_model.get_embedding(
                [audio_filepath]  # Pass the audio_filepath as a list
            )

            # get_embedding returns a list of tensors, one for each input file.
            # Since we passed a single file, we take the first element.
            # This will be a torch.Tensor
            speaker_emb_tensor = embeddings_list[0]
            speaker_emb_np = speaker_emb_tensor.cpu().numpy()  # Convert to NumPy array
            return speaker_emb_np

        except Exception as e:
            print(f"Error generating speaker embedding: {e}")
            traceback.print_exc()  # Use imported traceback
            return None

    def verify_speaker(self, new_embedding: np.ndarray, registered_embeddings_dir: str, threshold: float = 0.5) -> Tuple[bool, Optional[str], float]:
        """
        Compares a `new_embedding` against all registered speaker embeddings in the
        specified directory to find the best match.
        Returns (is_match, best_match_id, best_score).
        The best_match_id is the speaker ID from the filename (e.g., 'speaker_1').
        """
        self._check_initialized()
        if self.speaker_verification_model is None:
            print("Error: Speaker verification model not loaded. Cannot verify speaker.")
            return False, None, 0.0

        if not os.path.isdir(registered_embeddings_dir):
            print(
                f"Error: Registered embeddings directory not found: {registered_embeddings_dir}")
            return False, None, 0.0

        best_score = 0.0
        best_match_id = None

        new_embedding_tensor = torch.from_numpy(
            new_embedding).float().to(self.device).unsqueeze(0)

        # Iterate through saved embeddings
        for filename in os.listdir(registered_embeddings_dir):
            if filename.endswith("_embedding.pkl"):
                filepath = os.path.join(registered_embeddings_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        registered_embedding = pickle.load(f)

                    # Ensure registered_embedding is a numpy array and flatten if needed
                    if isinstance(registered_embedding, torch.Tensor):
                        registered_embedding = registered_embedding.cpu().numpy()
                    registered_embedding_np = registered_embedding.flatten()

                    if registered_embedding_np.ndim == 1 and new_embedding.ndim == 1 and \
                       registered_embedding_np.shape[0] == new_embedding.shape[0]:

                        registered_embedding_tensor = torch.from_numpy(
                            registered_embedding_np).float().to(self.device).unsqueeze(0)

                        # Calculate cosine similarity
                        similarity = F.cosine_similarity(
                            new_embedding_tensor, registered_embedding_tensor)
                        score = similarity.item()  # Convert to Python float

                        if score > best_score:
                            best_score = score
                            # Extract user ID from filename (e.g., "fastpitch_speaker_1_embedding.pkl" -> "fastpitch_speaker_1")
                            best_match_id = filename.replace(
                                "_embedding.pkl", "")

                except Exception as e:
                    print(
                        f"Warning: Could not load or process embedding from {filename}: {e}")
                    traceback.print_exc()
                    continue

        is_match = best_score >= threshold
        return is_match, best_match_id, best_score

    def authenticate_user(self, audio_filepath: str, registered_embeddings_dir: str, authentication_threshold: float = 0.5) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Authenticates a user based on their voice.
        Returns (user_id, speaker_embedding_np) if authenticated, else (None, None).
        Note: The `registered_embeddings_dir` is passed in from the caller,
              as AudioManager itself doesn't manage the registry.
        """
        self._check_initialized()
        if self.speaker_verification_model is None:
            print("Error: Speaker verification model not loaded. Cannot authenticate.")
            return None, None

        if not os.path.exists(registered_embeddings_dir):
            print(
                f"Error: Registered embeddings directory not found: {registered_embeddings_dir}")
            return None, None

        user_embedding = self.generate_speaker_embedding(audio_filepath)
        if user_embedding is None:
            print("Failed to generate embedding for authentication.")
            return None, None

        # This method calls verify_speaker, which is part of AudioManager
        is_match, best_match_id, best_score = self.verify_speaker(
            new_embedding=user_embedding,
            registered_embeddings_dir=registered_embeddings_dir,
            threshold=authentication_threshold
        )

        if is_match:
            print(
                f"User '{best_match_id}' successfully authenticated with score {best_score:.4f}.")
            # Return the matched ID and the new embedding
            return best_match_id, user_embedding
        else:
            print(
                f"Authentication failed. Best match was '{best_match_id}' with score {best_score:.4f} (below threshold {authentication_threshold}).")
            return None, None  # No match, return None for ID and embedding

    def synthesize_speech(self, text: str,
                          speaker_id: int = 0,  # Defaulted to 0 as per previous request
                          output_filename="response.wav", output_dir=None) -> Optional[str]:
        """
        Synthesizes speech from text using NeMo TTS models, using speaker_id
        to specify the speaker.
        """
        self._check_initialized()

        if self.tts_model is None or self.vocoder_model is None:
            print("Error: TTS models are not loaded. Cannot synthesize speech.")
            return None

        if output_dir is None:
            output_dir = self.audio_records_dir

        try:
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, output_filename)

            parsed_text = self.tts_model.parse(text)
            parsed_text = parsed_text.to(self.device)

            spectrogram_args = {"tokens": parsed_text}

            # Inspect the signature of the `generate_spectrogram` method
            gen_spec_signature = inspect.signature(
                self.tts_model.generate_spectrogram)
            gen_spec_params = gen_spec_signature.parameters

            # Check if the model's generate_spectrogram method accepts a 'speaker' parameter
            accepts_speaker_tensor_arg = 'speaker' in gen_spec_params

            if accepts_speaker_tensor_arg:
                # Convert the integer speaker_id to a torch.tensor as shown in the tutorial
                speaker_tensor = torch.tensor(
                    [speaker_id]).long().to(device=self.device)
                spectrogram_args["speaker"] = speaker_tensor
                print(
                    f"DEBUG: Using speaker ID {speaker_id} passed as a tensor for 'speaker' parameter.")
            else:
                # This branch would typically mean it's a single-speaker model or
                # expects speaker info differently, so we proceed without passing 'speaker'.
                print("INFO: TTS model's generate_spectrogram does not appear to accept a 'speaker' tensor parameter. Proceeding without explicit speaker parameter.")

            with torch.no_grad():
                # Generate spectrogram
                spectrogram = self.tts_model.generate_spectrogram(
                    **spectrogram_args)
                # Convert spectrogram to audio waveform
                audio = self.vocoder_model(spec=spectrogram)

            audio = audio.cpu().numpy()

            # Handle potential multi-dimensional audio output from vocoder for soundfile compatibility
            if audio.ndim == 3:
                audio = audio.squeeze(0)
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)
                elif audio.ndim == 2 and audio.shape[0] > 1:
                    audio = audio.T
            elif audio.ndim == 2 and audio.shape[0] == 1:
                audio = audio.squeeze(0)

            if audio.ndim > 2 or audio.ndim == 0:
                print(
                    f"Error: Final audio array has unsupported shape: {audio.shape}. Expected 1D or 2D (for sf.write).")
                return None

            # Normalize audio for playback/saving
            audio_norm = audio * (32767 / max(0.01, np.max(np.abs(audio))))
            audio_norm = audio_norm.astype(np.int16)

            # Save the synthesized speech to a file
            sf.write(output_filepath, audio_norm, config.TTS_SAMPLE_RATE)
            print(f"Speech synthesized and saved to '{output_filepath}'")

            # Play the synthesized speech
            sd.play(audio_norm, config.TTS_SAMPLE_RATE)
            sd.wait()
            return output_filepath

        except Exception as e:
            print(f"An error occurred during speech synthesis: {e}")
            traceback.print_exc()
            return None

    def set_global_speaker_id(self, speaker_id: int):
        self.global_speaker_id = speaker_id
        print(f"NOTICE: global speaker ID was set to {speaker_id}")


# Global instance of AudioManager
# This ensures that 'audio_manager' imported elsewhere always refers to this single instance.
audio_manager = AudioManager()
