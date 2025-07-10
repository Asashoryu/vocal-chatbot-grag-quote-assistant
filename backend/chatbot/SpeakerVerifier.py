# backend/chatbot/speaker_verifier.py

import os
import pickle as pkl
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import traceback
import operator
import torch.nn.functional as F
from typing import Optional, List, Tuple

import config


class SpeakerVerifier:
    """
    Manages speaker embedding extraction and speaker verification operations.
    It utilizes the AudioManager singleton for audio-related output (print and speak).
    """

    def __init__(self, model, audio_manager_instance=None):
        """
        Initializes the SpeakerVerifier with the NeMo speaker verification model.
        Configuration parameters are sourced from config.py and AudioManager.

        Args:
            model: The loaded NeMo speaker verification model.
            audio_manager_instance: An instance of AudioManager for print_and_speak.
                                    Passing it explicitly is better for dependency management.
        """
        print(
            f"DEBUG: SpeakerVerifier.__init__ received model type: {type(model)}")

        self.model = model
        self.target_sample_rate = config.SAMPLE_RATE
        self.embedding_output_dir = config.EMBEDDING_OUTPUT_DIR
        self.verification_threshold = config.VERIFICATION_THRESHOLD
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Assign the AudioManager instance
        self.audio_manager = audio_manager_instance
        if self.audio_manager is None:
            print("WARNING: AudioManager instance not provided to SpeakerVerifier. "
                  "print_and_speak functionality will be unavailable via this class.")

        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            # Internal WARNING, not for user conversation
            print(
                "WARNING: Speaker verification model not provided during SpeakerVerifier initialization.")

        # Ensure the embedding output directory exists
        os.makedirs(self.embedding_output_dir, exist_ok=True)

    def _speak_info(self, text: str, speaker_id: Optional[int] = None):
        """Helper to use AudioManager's print_and_speak if available."""
        if self.audio_manager:
            self.audio_manager.print_and_speak(text, speaker_id)
        else:
            # Fallback to print if AudioManager is not set
            print(text)

    def extract_speaker_embedding(self, audio_filepath: str, embedding_output_filename="speaker_embedding.pkl", save_embedding: bool = True) -> Optional[np.ndarray]:
        """
        Generates a speaker embedding from an audio file using the provided NeMo model
        and optionally saves the embedding. Handles resampling to target_sample_rate if needed.

        Args:
            audio_filepath (str): Path to the audio file.
            embedding_output_filename (str): Name of the file to save the embedding (if save_embedding is True).
            save_embedding (bool): Whether to save the extracted embedding to a file.

        Returns:
            np.ndarray: The extracted speaker embedding, or None if an error occurs.
        """
        if self.model is None:
            print(
                "Error: Speaker verification model is not loaded. Cannot extract embedding.")
            return None
        if not os.path.exists(audio_filepath):
            print(
                f"Error: Audio file not found at {audio_filepath}. Cannot extract embedding.")
            return None

        try:
            # Load the audio file as a waveform tensor using torchaudio
            waveform, sample_rate = torchaudio.load(audio_filepath)

            # Ensure the audio is mono (single channel) and has a batch dimension [1, num_samples]
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                # If multi-channel, take the mean across channels and keep dim [1, num_samples]
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.ndim == 1:
                # If 1D, add a channel dimension [1, num_samples]
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(
                    f"Unexpected audio waveform shape after initial processing: {waveform.shape}. Expected [1, samples] or [samples].")

            # Move the waveform to the correct device BEFORE resampling
            waveform = waveform.to(self.device)

            # Resample if necessary (NeMo models typically expect 16000 Hz)
            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                ).to(self.device)  # Resampler itself is already on device
                # Now, both waveform and resampler are on the same device
                waveform = resampler(waveform)

            # The length must be torch.long dtype
            audio_lengths = torch.tensor(
                [waveform.shape[1]], dtype=torch.long).to(self.device)

            # Check for silent audio after processing
            max_amplitude = torch.max(torch.abs(waveform)).item()
            # Threshold for considering "silent"
            if max_amplitude < 0.001:
                print(f"\nAudio file '{os.path.basename(audio_filepath)}' appears to be silent or extremely quiet after processing. This may lead to poor embedding quality. Please ensure the audio has clear speech.")

            with torch.no_grad():
                with torch.autocast(device_type=self.device, enabled=self.device == 'cuda'):
                    # Call the model's forward method. This typically returns (logits, embeddings).
                    # 'input_signal' and 'input_signal_length' are standard arguments for ASR/Speaker models.
                    logits, embs = self.model.forward(
                        input_signal=waveform, input_signal_length=audio_lengths
                    )
                    embedding = embs.cpu().detach().numpy().squeeze()

            # L2 normalization (usually done by NeMo, but good to ensure)
            raw_embs_l2_norm = np.linalg.norm(embedding)
            if raw_embs_l2_norm < 1e-6:  # Check if norm is essentially zero
                print(
                    f"WARNING: L2 norm of the extracted embedding for '{os.path.basename(audio_filepath)}' is zero or very close to zero. This indicates the model produced a silent/empty embedding. It will not be normalized.")
            else:
                # Normalize if not already (or if it's too small)
                embedding = embedding / raw_embs_l2_norm

            if save_embedding:
                embedding_filepath = os.path.join(
                    self.embedding_output_dir, embedding_output_filename)
                with open(embedding_filepath, 'wb') as f:
                    pkl.dump(embedding, f)
                # Internal status
                print(f"Embedding saved: {embedding_filepath}")

            return embedding

        except Exception as e:
            print(f"An error occurred during embedding extraction: {e}")
            # Print full traceback for debugging
            traceback.print_exc()
            print("Please ensure: 1. The audio file is valid and readable. 2. The NeMo model loaded successfully without errors.")
            return None

    def list_registered_speakers(self) -> List[str]:
        """
        Lists the inferred IDs (based on filename) of all registered speaker embeddings
        in the configured embedding output directory.

        Returns:
            List[str]: A sorted list of speaker IDs.
        """
        speaker_ids = []
        if not os.path.isdir(self.embedding_output_dir):
            print(
                f"WARNING: Embedding directory not found: {self.embedding_output_dir}")
            return []

        for filename in os.listdir(self.embedding_output_dir):
            if filename.endswith(('.pkl', '.npy')):
                # Assuming the ID is the filename without '_embedding' suffix and extension
                speaker_id = os.path.splitext(filename)[0]
                if speaker_id.endswith("_embedding"):
                    speaker_id = speaker_id[:-len("_embedding")]
                speaker_ids.append(speaker_id)
        # Return unique and sorted IDs
        return sorted(list(set(speaker_ids)))

    def verify_speaker_against_folder_embeddings(self, new_voice_audio_path: str, registered_embeddings_folder: str = None) -> Tuple[bool, Optional[str], Optional[float], Optional[np.ndarray], List[dict]]:
        """
        Verifies a new voice against all pre-saved embeddings in a given folder,
        orders the results by similarity, and reports the best match.

        Args:
            new_voice_audio_path (str): Path to the audio file of the new voice to be verified.
            registered_embeddings_folder (str, optional): Path to the folder containing
                                                          pre-saved speaker embeddings.
                                                          Defaults to self.embedding_output_dir.

        Returns:
            tuple: (is_verified, best_match_filename, best_score, best_matching_embedding_data, all_results_sorted)
                   - is_verified (bool): True if a match above threshold is found, False otherwise.
                   - best_match_filename (str or None): Filename (or inferred ID) of the best matching registered speaker.
                   - best_score (float or None): Cosine similarity score of the best match.
                   - best_matching_embedding_data (np.ndarray or None): The actual numpy array of the best matching registered embedding.
                   - all_results_sorted (list): List of all comparison results, sorted by score,
                                                each item is {'filename': str, 'score': float, 'is_match': bool}.
        """
        if self.model is None:
            self._speak_info(
                "Error at Speaker verification. Speaker verification model is not loaded. Cannot perform verification.", speaker_id=0)  # Changed None to speaker_id=0
            return False, None, None, None, []

        if registered_embeddings_folder is None:
            registered_embeddings_folder = self.embedding_output_dir

        if not os.path.exists(new_voice_audio_path):
            self._speak_info(
                f"Error: New voice audio file not found at {new_voice_audio_path}. Cannot perform verification.", speaker_id=0)  # Changed None to speaker_id=0
            return False, None, None, None, []
        if not os.path.isdir(registered_embeddings_folder):
            self._speak_info(
                f"Error: Registered embeddings folder not found at {registered_embeddings_folder}. Please ensure the folder exists and contains .pkl or .npy embedding files.", speaker_id=0)  # Changed None to speaker_id=0
            return False, None, None, None, []

        print(f"\n--- Starting Speaker Verification Against Registered Embeddings ---")
        print(
            f"Verifying '{os.path.basename(new_voice_audio_path)}' against embeddings in '{registered_embeddings_folder}'...")

        # Extract embedding for the new voice
        print("\nExtracting embedding for the new voice to be verified...")
        new_voice_embedding = self.extract_speaker_embedding(
            audio_filepath=new_voice_audio_path,
            save_embedding=False  # Don't save this temporary embedding
        )
        if new_voice_embedding is None:
            self._speak_info(
                "Error: Failed to extract embedding for the new voice. Aborting verification.", speaker_id=0)
            return False, None, None, None, []

        # Ensure embedding is 2D for cosine_similarity (1, embedding_dim)
        new_voice_embedding_tensor = new_voice_embedding.reshape(1, -1)

        all_results = []

        # Variables to store the best overall match details
        best_overall_score = -1.0
        best_overall_match_filename = None
        best_match_exceeds_threshold = False
        best_matching_embedding_data = None

        # Iterate through registered embeddings
        registered_files = [f for f in os.listdir(
            registered_embeddings_folder) if f.endswith(('.pkl', '.npy'))]

        if not registered_files:
            print(
                f"No .pkl or .npy embedding files found in '{registered_embeddings_folder}'.")
            return False, None, None, None, []

        print(
            f"\nFound {len(registered_files)} registered embeddings. Comparing...")

        for filename in registered_files:
            filepath = os.path.join(registered_embeddings_folder, filename)
            try:
                # Handle both .pkl and .npy if necessary
                if filename.endswith('.pkl'):
                    with open(filepath, 'rb') as f:
                        registered_embedding = pkl.load(f)
                elif filename.endswith('.npy'):
                    registered_embedding = np.load(filepath)
                else:
                    print(
                        f"Skipping '{filename}': Unsupported embedding file type.")
                    continue

                if not isinstance(registered_embedding, np.ndarray) or registered_embedding.ndim != 1:
                    print(
                        f"Skipping '{filename}': Invalid embedding format or shape. Expected 1D numpy array.")
                    continue

                registered_embedding_tensor = registered_embedding.reshape(
                    1, -1)

                score = F.cosine_similarity(
                    torch.from_numpy(new_voice_embedding_tensor).float().to(
                        self.device),
                    torch.from_numpy(registered_embedding_tensor).float().to(
                        self.device)
                ).item()
                is_same_speaker = score >= self.verification_threshold

                all_results.append(
                    {'filename': filename, 'score': score, 'is_match': is_same_speaker})

                # Only update best_match_exceeds_threshold if score meets threshold
                if is_same_speaker and score > best_overall_score:
                    best_overall_score = score
                    best_overall_match_filename = filename
                    best_match_exceeds_threshold = True
                    best_matching_embedding_data = registered_embedding.copy()  # Store a copy

            except Exception as e:
                print(
                    f"Error processing '{filename}': {e}. Skipping this file.")
                traceback.print_exc()
                continue

        # Sort results by similarity score (descending)
        all_results_sorted = sorted(
            all_results, key=operator.itemgetter('score'), reverse=True)

        # If no match exceeded threshold, but there were results, take the absolute best score regardless of threshold
        if not best_match_exceeds_threshold and all_results_sorted:
            # The actual best score (might be below threshold)
            highest_score_overall_below_threshold = all_results_sorted[0]['score']
            highest_score_filename_below_threshold = all_results_sorted[0]['filename']

            # If best_overall_score is still -1.0, and there are results, set it to the absolute highest
            if best_overall_score == -1.0 and highest_score_overall_below_threshold > -1.0:
                best_overall_score = highest_score_overall_below_threshold
                best_overall_match_filename = highest_score_filename_below_threshold
                # Load the embedding for this highest score below threshold
                best_match_filepath = os.path.join(
                    registered_embeddings_folder, highest_score_filename_below_threshold)
                if highest_score_filename_below_threshold.endswith('.pkl'):
                    with open(best_match_filepath, 'rb') as f:
                        best_matching_embedding_data = pkl.load(f)
                elif highest_score_filename_below_threshold.endswith('.npy'):
                    best_matching_embedding_data = np.load(best_match_filepath)

        # Final Verification Summary (only these are spoken to the user)
        print("\n--- Final Verification Summary ---")
        if best_match_exceeds_threshold:
            self._speak_info(
                f"SUCCESS: The new voice is identified as '{best_overall_match_filename}'!", speaker_id=0)
            self._speak_info(
                f"           Similarity Score is {best_overall_score:.4f} (Above the threshold that is {self.verification_threshold:.4f})", speaker_id=0)
        elif all_results_sorted:
            # Get the highest score even if it's below the threshold
            highest_score_overall = all_results_sorted[0]['score']
            # highest_score_filename = all_results_sorted[0]['filename']
            self._speak_info(
                f"NO MATCH: The new voice is likely DIFFERENT from all registered speakers.", speaker_id=0)
            self._speak_info(
                f"           Highest similarity found at score {highest_score_overall:.4f} (Below the threshold that is {self.verification_threshold:.4f}).", speaker_id=0)  # Changed None to speaker_id=0
        else:
            print(
                "No registered embeddings were processed, or an error occurred. Cannot verify speaker.")

        return best_match_exceeds_threshold, best_overall_match_filename, best_overall_score, best_matching_embedding_data, all_results_sorted

    def get_closest_fastpitch_speaker_id_from_embedding(self, user_embedding_np: np.ndarray) -> Optional[int]:
        if user_embedding_np is None:
            self._speak_info(
                "User embedding is None for FastPitch matching. Skipping.", speaker_id=0)
            return None

        fastpitch_embeddings_dir = config.FASTPITCH_EMBEDDINGS_DIR
        if not os.path.isdir(fastpitch_embeddings_dir):
            self._speak_info(
                f"WARNING: FastPitch embeddings directory not found at {fastpitch_embeddings_dir}.", speaker_id=0)
            return None

        # Prepare user embedding tensor
        user_emb = torch.from_numpy(user_embedding_np).float().to(self.device)
        user_emb = user_emb.reshape(1, -1)  # shape (1, 384)

        max_similarity = -1.0
        closest_fastpitch_id = None

        for fn in os.listdir(fastpitch_embeddings_dir):
            if not (fn.endswith('.pkl') or fn.endswith('.npy')):
                continue

            base, _ = os.path.splitext(fn)
            parts = base.split('_', 1)
            if not (parts[0].isdigit() and "_fastpitch_speaker__embedding" in base):
                continue

            speaker_id = int(parts[0])
            fp = os.path.join(fastpitch_embeddings_dir, fn)

            try:
                if fn.endswith('.pkl'):
                    with open(fp, 'rb') as f:
                        try:
                            obj = torch.load(f, map_location='cpu')
                        except Exception:
                            f.seek(0)
                            obj = pkl.load(f)

                        if isinstance(obj, torch.Tensor):
                            current_np = obj.numpy()
                        elif isinstance(obj, np.ndarray):
                            current_np = obj
                        else:
                            # unsupported type
                            continue

                # ensure correct shape
                current_np = current_np.reshape(1, -1)
                fp_emb = torch.from_numpy(current_np).float().to(self.device)

                similarity = F.cosine_similarity(user_emb, fp_emb).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_fastpitch_id = speaker_id

            except Exception as e:
                print(f"WARNING: Error processing '{fn}': {e}")
                traceback.print_exc()
                continue

        if closest_fastpitch_id is not None:
            print(
                f"DEBUG: Closest FastPitch speaker ID: {closest_fastpitch_id}, similarity: {max_similarity:.4f}")
        else:
            self._speak_info(
                "No FastPitch speaker embeddings found or processed for matching.", speaker_id=0)

        return closest_fastpitch_id
