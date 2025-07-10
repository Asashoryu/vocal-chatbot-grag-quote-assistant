import os
import asyncio
import torch
import pickle
import traceback
import soundfile as sf

import config

from AudioManager import audio_manager
from ModelLoaderManager import model_loader_manager

from CommandLineMenu import CommandLineMenu

from Neo4jConnector import Neo4jConnector


def ensure_fastpitch_embeddings_exist():
    """
    Generates an audio saying "The quick brown fox jumps over the lazy dog."
    for specific TTS speaker IDs, extracts their speaker embeddings,
    and saves them to config.FASTPITCH_EMBEDDINGS_DIR.
    Existing embeddings are skipped.

    This function now explicitly checks for a multi-speaker FastPitch model
    and processes a defined list of speaker IDs.
    """
    print("\n--- Generating and Saving FastPitch Speaker Embeddings from Audio ---")

    # Ensure the embedding directory exists
    os.makedirs(config.FASTPITCH_EMBEDDINGS_DIR, exist_ok=True)
    # Ensure a temporary directory for generated audio exists
    temp_audio_gen_dir = os.path.join(
        config.AUDIO_RECORDS_DIR, "fastpitch_audio_gen_192")
    os.makedirs(temp_audio_gen_dir, exist_ok=True)

    if audio_manager.tts_model is None or audio_manager.vocoder_model is None:
        print("ERROR: TTS and/or Vocoder models are not loaded in AudioManager. Cannot generate audio for embeddings.")
        return

    speaker_verification_model = audio_manager.speaker_verification_model
    if speaker_verification_model is None:
        print("ERROR: Speaker verification model not loaded in AudioManager. Cannot extract embeddings from audio.")
        print("Please ensure 'speaker_verification_model' is properly initialized and loaded in AudioManager.")
        return

    tts_model = audio_manager.tts_model

    # Check if the model is a multi-speaker FastPitch model
    is_multispeaker_fastpitch = False
    if hasattr(tts_model, 'fastpitch') and hasattr(tts_model.fastpitch, 'speaker_emb') and tts_model.fastpitch.speaker_emb is not None:
        is_multispeaker_fastpitch = True
    elif hasattr(tts_model, 'num_speakers') and tts_model.num_speakers > 1:
        is_multispeaker_fastpitch = True

    if not is_multispeaker_fastpitch:
        print("INFO: The loaded TTS model is not identified as a multi-speaker FastPitch model. Exiting embedding generation.")
        return

    text_to_synthesize = "The quick brown fox jumps over the lazy dog."

    # Create the combined list of speaker IDs to process, start with IDs from 0 to 19
    speaker_ids_initial_range = list(range(0, 20))  # Generates [0, 1, ..., 19]

    # Add the specific additional speaker IDs from the imported FastPitch Multispeaker
    additional_speaker_ids = [
        92, 6097, 9017, 6670, 6671, 8051, 9136, 11614, 11697, 12787
    ]

    # Combine the ids using a set to ensure uniqueness, then convert back to a sorted list
    speaker_ids_to_process = sorted(
        list(set(speaker_ids_initial_range + additional_speaker_ids)))
    # --- END MODIFIED LOGIC ---

    print(
        f"Attempting to generate audio and extract embeddings for {len(speaker_ids_to_process)} specific FastPitch speakers.")
    print(f"Speaker IDs to process: {speaker_ids_to_process}")

    for speaker_id in speaker_ids_to_process:
        embedding_filepath = os.path.join(
            config.FASTPITCH_EMBEDDINGS_DIR, f"{speaker_id}_fastpitch_speaker__embedding.pkl"
        )

        if os.path.exists(embedding_filepath):
            print(
                f"FastPitch speaker {speaker_id} embedding already exists. Skipping.")
            continue

        print(f"Processing speaker ID {speaker_id}...")
        temp_audio_filename = f"temp_fastpitch_speaker_{speaker_id}.wav"
        temp_audio_filepath = os.path.join(
            temp_audio_gen_dir, temp_audio_filename)

        try:
            generated_audio_path = audio_manager.synthesize_speech(
                text=text_to_synthesize,
                speaker_id=speaker_id,
                output_filename=temp_audio_filename,
                output_dir=temp_audio_gen_dir
            )

            if generated_audio_path is None:
                print(
                    f"WARNING: Could not generate audio for speaker ID {speaker_id}. Skipping embedding extraction.")
                continue

            extracted_embedding = speaker_verification_model.get_embedding(
                generated_audio_path
            )

            if isinstance(extracted_embedding, torch.Tensor):
                extracted_embedding = extracted_embedding.cpu().numpy()

            if extracted_embedding.ndim > 1 and extracted_embedding.shape[0] == 1:
                extracted_embedding = extracted_embedding.squeeze(0)

            if extracted_embedding is None:
                print(
                    f"WARNING: Could not extract embedding from audio for speaker ID {speaker_id}. Skipping save.")
                continue

            with open(embedding_filepath, 'wb') as f:
                pickle.dump(extracted_embedding, f)
            print(
                f"Saved FastPitch speaker {speaker_id} embedding to {embedding_filepath}.")

        except IndexError:
            print(
                f"INFO: Speaker ID {speaker_id} is out of range for the loaded TTS model's capabilities. Skipping this ID."
            )
            continue
        except Exception as e:
            print(f"ERROR: Failed to process speaker {speaker_id}: {e}")
            traceback.print_exc()

    print("--- FastPitch Speaker Embedding Generation from Audio Complete ---")


async def main():
    """
    Main entry point for the vocal chatbot application.
    Handles initialization of models, audio components, Neo4j connection,
    and starts the interactive command-line menu.
    """
    print("Starting application setup...")

    # Ensure necessary directories are created at startup
    try:
        os.makedirs(config.AUDIO_RECORDS_DIR, exist_ok=True)
        os.makedirs(config.EMBEDDING_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.FASTPITCH_EMBEDDINGS_DIR, exist_ok=True)
        print(
            f"Ensured directories exist: {config.AUDIO_RECORDS_DIR}, {config.EMBEDDING_OUTPUT_DIR}, {config.FASTPITCH_EMBEDDINGS_DIR}")
    except Exception as e:
        print(f"FATAL ERROR: Could not create necessary directories: {e}")
        print("Please check directory paths and permissions. Exiting application.")
        return

    # Initialize ModelLoaderManager ONCE
    print("\n--- Initializing ModelLoaderManager and loading models ---")
    try:
        model_loader_manager.initialize_all_models()
        print("ModelLoaderManager initialized and models loaded successfully.")
    except Exception as e:
        print(
            f"\nFATAL ERROR: Failed to initialize ModelLoaderManager or load models: {e}")
        print("Please ensure NeMo dependencies are met, models are available, and check internet connection for downloads.")
        print("Exiting application.")
        return

    # Initialize AudioManager ONCE with models from ModelLoaderManager
    print("\n--- Initializing AudioManager ---")
    try:
        audio_manager.initialize(
            asr_model=model_loader_manager.asr_model,
            tts_model=model_loader_manager.tts_model,
            vocoder_model=model_loader_manager.vocoder_model,
            sample_rate=config.SAMPLE_RATE,
            recording_duration_seconds=config.RECORDING_DURATION_SECONDS,
            audio_records_dir=config.AUDIO_RECORDS_DIR,
            use_tts_for_answers_flag=config.USE_TTS_FOR_ANSWERS,
            speaker_verification_model=model_loader_manager.verification_model
        )
        print("AudioManager initialized successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to initialize AudioManager: {e}")
        print("Please check audio device setup, NeMo model paths, and AudioManager configuration.")
        print("Exiting application.")
        return

    # Perform an initial Neo4j connection test
    print("\n--- Attempting initial Neo4j connection test ---")
    test_neo4j_connector = None
    try:
        test_neo4j_connector = Neo4jConnector(
            config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD
        )
        if not test_neo4j_connector.connect():
            print("\nFATAL ERROR: Could not connect to Neo4j database.")
            print(
                f"Please check your Neo4j server status, URI ({config.NEO4J_URI}), username, and password.")
            print("Exiting application.")
            return
        else:
            print("Neo4j database connection successful.")
    except Exception as e:
        print(
            f"\nFATAL ERROR: An unexpected error occurred during Neo4j connection test: {e}")
        print("Exiting application.")
        return
    finally:
        # Ensure the test connection is closed, regardless of success or failure
        if test_neo4j_connector:
            test_neo4j_connector.close()
            print("Neo4j test connection closed.")

    print("\n--- Starting command line menu ---")
    try:
        menu = CommandLineMenu(
            neo4j_uri=config.NEO4J_URI,
            neo4j_username=config.NEO4J_USERNAME,
            neo4j_password=config.NEO4J_PASSWORD,
            entity_index_map=config.ENTITY_INDEX_MAP,
            speaker_verification_model=model_loader_manager.verification_model
        )
        # await menu.start_interactive_menu()
        await menu.start_interactive_menu()
    except Exception as e:
        print(
            f"\nFATAL ERROR: An error occurred during CommandLineMenu operation: {e}")
        print("Exiting application.")
    finally:
        print("\nApplication finished.")

if __name__ == "__main__":
    asyncio.run(main())
