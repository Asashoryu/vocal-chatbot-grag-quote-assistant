# backend/chatbot/ModelLoaderManager.py

import os
import torch
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
import traceback


class ModelLoaderManager:
    """
    Manages the loading of all NeMo models. Implemented as a singleton
    to ensure models are loaded only once and are globally accessible.
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        """Ensures that only one instance of ModelLoaderManager is ever created."""
        if cls._instance is None:
            cls._instance = super(ModelLoaderManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the ModelLoaderManager's internal state. This runs only once due to __new__.
        Sets up initial state for models.
        """
        if not self._is_initialized:
            self._verification_model = None
            self._asr_model = None
            self._tts_model = None
            self._vocoder_model = None
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(
                f"ModelLoaderManager initialized. Detected device: {self._device.upper()}")
            self._is_initialized = True

    def _check_initialized(self):
        """Internal helper to ensure models are loaded before access."""
        if not self._is_initialized:
            raise RuntimeError(
                "ModelLoaderManager not initialized. Call initialize() first.")

    def initialize_all_models(self):
        """
        Loads all required NeMo models. This method should be called once
        at the application's startup.
        """
        print("\n--- Starting Model Loading Process ---")
        self.load_speaker_verification_model()
        self.load_asr_model()
        self.load_fastpitch_model()
        self.load_hifigan_model()
        print("--- All Models Loading Process Completed ---")

    def load_speaker_verification_model(self):
        print("\n--- Starting TitaNet-Large speaker verification model ---")
        if self._verification_model is not None:
            print("Speaker Verification model already loaded.")
            return self._verification_model

        print(
            "Attempting to load pre-trained Speaker Verification model (TitaNet-Large)...")
        try:
            self._verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet_large")
            self._verification_model.eval()
            self._verification_model = self._verification_model.to(
                self._device)
            print("SUCCESS: successfully loaded pre-trained TitaNet-Large model.")
        except Exception as e:
            print(f"ERROR: Failed to load TitaNet model: {e}")
            traceback.print_exc()
            print("Speaker verification functionality may be limited.")
        return self._verification_model

    def load_asr_model(self):
        print("\n--- Starting FastConformer-Large ASR model ---")
        if self._asr_model is not None:
            print("ASR model already loaded.")
            return self._asr_model

        print("\nAttempting to load ASR model (FastConformer-Large)...")
        try:
            self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/stt_en_fastconformer_transducer_large")
            self._asr_model = self._asr_model.to(self._device)
            self._asr_model.eval()
            print("SUCCESS: successfully loaded ASR model from HuggingFace.")
        except Exception as e:
            print(f"ERROR: Failed to load ASR model: {e}")
            traceback.print_exc()
            print("ASR functionality may be limited or unavailable.")
            self._asr_model = None
        print(
            f"DEBUG: _asr_model state after loading attempt: {self._asr_model is not None}")
        return self._asr_model

    def load_fastpitch_model(self):
        print("\n--- Starting FastPitch TTS model ---")
        if self._tts_model is not None:
            print("FastPitch TTS model already loaded.")
            return self._tts_model
        fastpitch_name = "tts_en_fastpitch_multispeaker"
        print(f"\nAttempting to load {fastpitch_name} model...")
        try:
            self._tts_model = nemo_tts.models.FastPitchModel.from_pretrained(
                model_name=fastpitch_name)
            self._tts_model.eval()
            self._tts_model = self._tts_model.to(self._device)
            print("Success: successfully loaded FastPitch TTS model.")

        except Exception as e:
            print(f"ERROR: Failed to load FastPitch TTS model: {e}")
            traceback.print_exc()
            print("FastPitch TTS functionality may be limited.")
            self._tts_model = None
        return self._tts_model

    def load_hifigan_model(self):
        print("\n--- Starting HiFi-GAN vocoder model ---")
        if self._vocoder_model is not None:
            print("HiFi-GAN vocoder model already loaded.")
            return self._vocoder_model

        print("\nAttempting to load HiFi-GAN vocoder model...")
        try:
            self._vocoder_model = nemo_tts.models.HifiGanModel.from_pretrained(
                model_name="tts_en_hifitts_hifigan_ft_fastpitch")
            self._vocoder_model.eval()
            self._vocoder_model = self._vocoder_model.to(self._device)
            print("Successfully loaded HiFi-GAN vocoder model.")
        except Exception as e:
            print(f"ERROR: Failed to load HiFi-GAN vocoder model: {e}")
            traceback.print_exc()
            print("HiFi-GAN vocoder functionality may be limited.")
            self._vocoder_model = None
        return self._vocoder_model

    @property
    def verification_model(self):
        self._check_initialized()
        return self._verification_model

    @property
    def asr_model(self):
        self._check_initialized()
        return self._asr_model

    @property
    def tts_model(self):
        self._check_initialized()
        return self._tts_model

    @property
    def vocoder_model(self):
        self._check_initialized()
        return self._vocoder_model


model_loader_manager = ModelLoaderManager()
