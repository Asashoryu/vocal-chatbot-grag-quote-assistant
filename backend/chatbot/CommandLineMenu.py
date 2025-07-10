# command_line_menu.py

import os
import soundfile as sf
import pickle as pkl
import shutil
import asyncio
import numpy as np
import traceback

import config
import torch

from AudioManager import audio_manager
from ModelLoaderManager import model_loader_manager
from SpeakerVerifier import SpeakerVerifier

from ConversationalNeo4jChatbot import ConversationalNeo4jChatbot


class CommandLineMenu:
    """
    Manages the command-line interface and interactions for the Vocal Chatbot System.
    """

    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str,
                 entity_index_map: dict, speaker_verification_model):
        self.running = True

        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.entity_index_map = entity_index_map

        self.speaker_verifier = SpeakerVerifier(
            model=speaker_verification_model,
            audio_manager_instance=audio_manager
        )
        print(f"\nDEBUG: CommandLineMenu: SpeakerVerifier initialized.")

        # Initialize core components
        self.neo4j_chatbot = ConversationalNeo4jChatbot(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            entity_index_map=self.entity_index_map,
            llm_call_function=None
        )

        # Mapping for spoken menu choices to numerical choices
        self.spoken_numbers_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "exit": "0", "register": "1",
            "authenticate": "2", "chat": "3", "query": "4",
            "select": "5", "list": "6", "delete": "7",
            "back": "back", "menu": "menu"
        }

    async def start_interactive_menu(self):
        """
        Starts the main interactive command-line menu loop.
        """
        # Initial check for model loading, though ModelLoaderManager handles errors internally
        if any(model is None for model in [
            model_loader_manager.verification_model,
            model_loader_manager.asr_model,
            model_loader_manager.tts_model,
            model_loader_manager.vocoder_model
        ]):
            print("\n**WARNING**: One or more required models could not be loaded. Some functionalities may be unavailable.")

        audio_manager.print_and_speak("Welcome to the Vocal Chatbot!")

        if audio_manager.use_tts_for_answers_flag:
            print("NOTICE: Text to speech for computer answers is currently **enabled**.")
        else:
            print(
                "NOTICE: Text to speech for computer answers is currently **disabled**.")

        while True:
            self._display_menu()
            await self._handle_menu_choice()
            await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting

    def _display_menu(self):
        """Prints the main menu options to the console."""
        print("\n" + "="*50)
        print("VOCAL CHATBOT SYSTEM MENU")
        print("="*50)
        print("1. Register a new speaker")
        print("2. Authenticate User (Voice Identification)")
        print("3. Start Voice Chat (Authenticated - Multi-Turn)")
        print("4. Conversational Neo4j Query (Unauthenticated - Multi-Turn)")
        # Renumbered options
        print("5. Select TTS Answer Voice from Registered Embedding")
        print("6. List all registered speaker embeddings")
        print("7. Delete a registered speaker embedding")
        print("0. Exit")
        print("="*50)

    async def _handle_menu_choice(self):
        """Handles the user's spoken menu choice."""
        audio_manager.print_and_speak(
            "Please speak your choice from the menu.")

        recorded_menu_audio_data = audio_manager.record_audio(
            prompt_message=f"Please speak your menu choice (e.g., 'one', 'two', 'exit') for {config.RECORDING_DURATION_SECONDS} seconds."
        )

        choice = ""
        transcribed_menu_choice = ""
        menu_choice_audio_filepath = None

        if recorded_menu_audio_data is not None:
            try:
                temp_audio_output_filename = "menu_choice_temp.wav"
                menu_choice_audio_filepath = os.path.join(
                    audio_manager.audio_records_dir, temp_audio_output_filename)

                sf.write(menu_choice_audio_filepath,
                         recorded_menu_audio_data, audio_manager.sample_rate)
                print(
                    f"Temporary audio saved to: {menu_choice_audio_filepath}")

                transcribed_menu_choice = audio_manager.transcribe_audio(
                    menu_choice_audio_filepath)

                if transcribed_menu_choice:
                    print(
                        f"The Transcription is: \"{transcribed_menu_choice}\"")
                else:
                    audio_manager.print_and_speak(
                        "I couldn't transcribe your speech for the menu choice. Please try again.")
                    return  # Go back to main loop for another attempt

            finally:
                if menu_choice_audio_filepath and os.path.exists(menu_choice_audio_filepath):
                    os.remove(menu_choice_audio_filepath)
                    print(
                        f"Cleaned up temporary menu audio: {menu_choice_audio_filepath}")
        else:
            audio_manager.print_and_speak(
                "No audio recorded for menu choice. Please try again.")
            return

        lower_transcribed_menu_choice = transcribed_menu_choice.lower()
        for word, digit in self.spoken_numbers_map.items():
            if word in lower_transcribed_menu_choice:
                choice = digit
                break

        if not choice and lower_transcribed_menu_choice.isdigit():
            choice = lower_transcribed_menu_choice.strip()

        if not choice:
            audio_manager.print_and_speak(
                "I didn't understand your choice. Please try speaking clearly a number from the menu.")
            return

        print(
            f"User's spoken choice (transcribed): {transcribed_menu_choice} -> Processed Choice: {choice}")

        await self._execute_menu_action(choice)

    async def _execute_menu_action(self, choice: str):
        """Executes the action corresponding to the given menu choice."""
        if choice == '1':
            await self._register_new_speaker()
        elif choice == '2':
            await self._authenticate_user()
        elif choice == '3':
            await self._start_voice_chat()
        elif choice == '4':
            await self._start_conversational_neo4j_query()
        elif choice == '5':
            await self._select_tts_voice_from_embedding()
        elif choice == '6':
            self._list_registered_speakers()
        elif choice == '7':
            self._delete_speaker_embedding()
        elif choice == '0':
            audio_manager.print_and_speak(
                "\nExiting Vocal Chatbot System. Goodbye!")
            exit()
        else:
            audio_manager.print_and_speak(
                "Invalid choice. Please enter a number from the menu.")

    async def _register_new_speaker(self):
        """
        Handles the 'Register a new speaker' menu option.
        Records audio, saves it permanently, extracts the speaker embedding,
        and saves the embedding to the configured directory.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to register a new user.")
        user_id = input(
            "Enter a unique ID for the speaker (e.g., 'Alice', 'Bob'): ").strip()
        if not user_id:
            audio_manager.print_and_speak(
                "Speaker ID cannot be empty. Returning to menu.")
            return

        # Ensure the user ID is valid for filenames and not 'clear'
        if user_id.lower() == 'clear':
            audio_manager.print_and_speak(
                "User ID cannot be 'clear'. Please choose a different ID.")
            return

        # Define the permanent path for the recorded registration audio
        reg_audio_filename = f"{user_id}_registration_audio.wav"
        reg_audio_filepath = os.path.join(
            config.AUDIO_RECORDS_DIR, reg_audio_filename)

        recorded_audio_data = audio_manager.record_audio(
            prompt_message=f"Please speak for {audio_manager.recording_duration_seconds} seconds to register '{user_id}'."
        )

        if recorded_audio_data is not None:
            try:
                # Save the recorded audio permanently to the designated directory
                sf.write(reg_audio_filepath, recorded_audio_data,
                         audio_manager.sample_rate)
                print(f"Recorded audio saved to: {reg_audio_filepath}")

                # Extract speaker verification embedding using SpeakerVerifier
                # The `save_embedding=True` flag in SpeakerVerifier's method
                # means it will handle saving the embedding to config.EMBEDDING_OUTPUT_DIR
                speaker_verification_embedding = self.speaker_verifier.extract_speaker_embedding(
                    audio_filepath=reg_audio_filepath,  # Use the permanently saved audio file
                    embedding_output_filename=f"{user_id}_embedding.pkl",
                    save_embedding=True
                )

                if speaker_verification_embedding is not None:
                    audio_manager.print_and_speak(
                        f"User '{user_id}' registered successfully!")
                else:
                    audio_manager.print_and_speak(
                        f"Failed to extract speaker embedding for '{user_id}'. User not fully registered.")
            except Exception as e:
                print(f"ERROR during speaker registration: {e}")
                traceback.print_exc()
                audio_manager.print_and_speak(
                    f"An error occurred during registration: {str(e)}")
        else:
            audio_manager.print_and_speak(
                "Audio recording failed. User not registered.")

    async def _authenticate_user(self):
        """
        Handles the 'Authenticate User' menu option.
        Authenticates the user and, if successful, sets the global TTS voice
        to the FastPitch speaker ID closest to the authenticated user's voice.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to authenticate the user.")

        recorded_auth_audio_data = audio_manager.record_audio(
            prompt_message=f"Please speak for {audio_manager.recording_duration_seconds} seconds for authentication."
        )

        temp_audio_filepath = None
        user_extracted_embedding = None

        if recorded_auth_audio_data is not None:
            try:
                temp_audio_output_filename = "voice_to_verify_temp.wav"
                temp_audio_filepath = os.path.join(
                    audio_manager.audio_records_dir, temp_audio_output_filename)
                sf.write(temp_audio_filepath, recorded_auth_audio_data,
                         audio_manager.sample_rate)
                print(
                    f"Temporary audio saved for authentication: {temp_audio_filepath}")

                # Extract the embedding of the new voice for authentication
                user_extracted_embedding = self.speaker_verifier.extract_speaker_embedding(
                    audio_filepath=temp_audio_filepath,
                    save_embedding=False  # We just need the embedding in memory for now
                )

                if user_extracted_embedding is None:
                    audio_manager.print_and_speak(
                        "Failed to extract voice embedding for authentication. Cannot authenticate.")
                    print("Error: Could not extract user voice embedding.")
                    return

                # Use the extracted embedding for verification (via the audio file path)
                is_identified, identified_user_filename, score, _, all_speaker_results = \
                    self.speaker_verifier.verify_speaker_against_folder_embeddings(
                        new_voice_audio_path=temp_audio_filepath
                    )

                recognized_user_id = "unknown"
                if is_identified:
                    # Remove '_embedding.pkl' or '_embedding.npy' from filename to get user ID
                    recognized_user_id = identified_user_filename.replace(
                        "_embedding.pkl", "").replace("_embedding.npy", "")
                    audio_manager.print_and_speak(
                        f"Authentication was successful!")
                    print(
                        f"Authentication successful! Identified User: {recognized_user_id}. Score: {score:.4f}")

                    # Find the closest FastPitch speaker ID based on the authenticated user's voice embedding
                    closest_fastpitch_id = self.speaker_verifier.get_closest_fastpitch_speaker_id_from_embedding(
                        user_extracted_embedding
                    )

                    if closest_fastpitch_id is not None:
                        audio_manager.set_global_speaker_id(
                            closest_fastpitch_id)
                        audio_manager.print_and_speak(
                            f"TTS voice set to FastPitch Speaker ID {closest_fastpitch_id} for personalization.")
                    else:
                        audio_manager.print_and_speak(
                            "Could not find a matching FastPitch speaker ID. Using default TTS voice (ID 0).")
                        audio_manager.set_global_speaker_id(
                            0)  # Fallback to default speaker 0

                else:
                    audio_manager.print_and_speak(
                        f"Authentication failed.")
                    # Get the user ID from the best match filename if available
                    best_match_id_display = identified_user_filename.replace('_embedding.pkl', '').replace(
                        '_embedding.npy', '') if identified_user_filename else 'none'
                    print(
                        f"Authentication failed. User not identified. Best match (if any) was {best_match_id_display} with score: {score:.4f}")
                    for result_dict in all_speaker_results:
                        print(
                            f" - {result_dict['filename'].replace('_embedding.pkl', '').replace('_embedding.npy', '')} | Score: {result_dict['score']:.4f}")

            finally:
                # Clean up the temporary audio file used for authentication
                if temp_audio_filepath and os.path.exists(temp_audio_filepath):
                    os.remove(temp_audio_filepath)
                    print(
                        f"Cleaned up temporary auth audio: {temp_audio_filepath}")
        else:
            audio_manager.print_and_speak(
                "No audio recorded for authentication.")

    async def _start_voice_chat(self):
        """
        Handles the 'Start Voice Chat (Authenticated - Multi-Turn)' menu option.
        Authenticates the user and then starts a multi-turn conversation,
        setting the TTS voice based on the authenticated user's voice by finding
        the closest FastPitch speaker ID.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to start an Authenticated Voice Chat.")
        audio_manager.print_and_speak("Please authenticate yourself first.")

        recorded_chat_auth_audio_data = audio_manager.record_audio(
            prompt_message=f"Speak for {audio_manager.recording_duration_seconds} seconds to authenticate for chat."
        )

        temp_auth_audio_filepath = None
        user_extracted_embedding = None

        if recorded_chat_auth_audio_data is not None:
            try:
                temp_auth_audio_output_filename = "chat_auth_voice_temp.wav"
                temp_auth_audio_filepath = os.path.join(
                    audio_manager.audio_records_dir, temp_auth_audio_output_filename)
                sf.write(temp_auth_audio_filepath,
                         recorded_chat_auth_audio_data, audio_manager.sample_rate)
                print(
                    f"Temporary audio saved for chat authentication: {temp_auth_audio_filepath}")

                user_extracted_embedding = self.speaker_verifier.extract_speaker_embedding(
                    audio_filepath=temp_auth_audio_filepath,
                    save_embedding=False
                )

                if user_extracted_embedding is None:
                    audio_manager.print_and_speak(
                        "Failed to extract voice embedding for authentication. Cannot start chat.")
                    print("Error: Could not extract user voice embedding.")
                    return

                is_identified, identified_user_filename, score, _, _ = \
                    self.speaker_verifier.verify_speaker_against_folder_embeddings(
                        new_voice_audio_path=temp_auth_audio_filepath
                    )

                if is_identified:
                    recognized_user_name = identified_user_filename.replace(
                        "_embedding.pkl", "").replace("_embedding.npy", "")
                    audio_manager.print_and_speak(
                        f"Authenticated as: {recognized_user_name}. Starting conversation...")
                    print(
                        f"Authentication successful. User: {recognized_user_name}. Score: {score:.4f}")

                    closest_fastpitch_id = self.speaker_verifier.get_closest_fastpitch_speaker_id_from_embedding(
                        user_extracted_embedding
                    )

                    if closest_fastpitch_id is not None:
                        audio_manager.set_global_speaker_id(
                            closest_fastpitch_id)
                        audio_manager.print_and_speak(
                            f"TTS voice set to FastPitch Speaker ID {closest_fastpitch_id} for personalization.")
                    else:
                        audio_manager.print_and_speak(
                            "Could not find a matching FastPitch speaker ID. Using default TTS voice (ID 0).")
                        audio_manager.set_global_speaker_id(0)

                    await self.neo4j_chatbot.start_conversation_loop(recognized_user_name)

                    audio_manager.print_and_speak(
                        f"Chat session for {recognized_user_name} ended.")
                else:
                    audio_manager.print_and_speak(
                        "Authentication failed. Cannot start chat. Returning to the menu.")
                    print("Authentication failed.")
                    return
            finally:
                if temp_auth_audio_filepath and os.path.exists(temp_auth_audio_filepath):
                    os.remove(temp_auth_audio_filepath)
                    print(
                        f"Cleaned up temporary auth audio: {temp_auth_audio_filepath}")
        else:
            audio_manager.print_and_speak(
                "Authentication audio not recorded. Cannot start chat.")

    async def _start_conversational_neo4j_query(self):
        """
        Handles the 'Conversational Neo4j Query' menu option.
        Starts a multi-turn, unauthenticated conversation.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to start an Unauthenticated Conversational Query.")
        await self.neo4j_chatbot.start_conversation_loop()

    async def _transcribe_audio_file(self):
        """Handles the 'Transcribe an audio file' menu option."""
        audio_manager.print_and_speak(
            "\nYou have chosen to transcribe with ASR an audio file.")
        input_audio_path = input(
            "Enter path to audio file for transcription: ").strip()
        if os.path.exists(input_audio_path):
            transcribed_text = audio_manager.transcribe_audio(input_audio_path)
            if transcribed_text:
                print(f"Transcription: {transcribed_text}")
                audio_manager.print_and_speak(
                    f"The transcription is: {transcribed_text}")
            else:
                audio_manager.print_and_speak(
                    "Failed to transcribe audio.")
        else:
            audio_manager.print_and_speak(
                f"Error: File not found at the path.")
            print(f"Error: File not found at {input_audio_path}.")

    async def _select_tts_voice_from_embedding(self):
        """
        Handles the 'Select TTS Answer Voice from FastPitch Embedding' menu option.
        Allows the user to choose a pre-trained FastPitch speaker ID
        and demonstrates it by speaking a user-provided text with the selected voice.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to select a default voice for the chatbot's answers from FastPitch models."
        )
        print("\n--- Available FastPitch Speaker IDs ---")

        available_speaker_ids = []
        if not os.path.isdir(config.FASTPITCH_EMBEDDINGS_DIR):
            audio_manager.print_and_speak(
                f"Error: FastPitch embeddings directory not found at {config.FASTPITCH_EMBEDDINGS_DIR}."
            )
            print("Please ensure FastPitch embeddings have been generated.")
            return

        # Collect available FastPitch speaker IDs from filenames like: 1_fastpitch_speaker__embedding.pkl
        for filename in os.listdir(config.FASTPITCH_EMBEDDINGS_DIR):
            if filename.endswith("__embedding.pkl") or filename.endswith("__embedding.npy"):
                try:
                    speaker_id_str = filename.split(
                        "_fastpitch_speaker__embedding")[0]
                    speaker_id = int(speaker_id_str)
                    available_speaker_ids.append(speaker_id)
                except ValueError:
                    print(
                        f"Warning: Could not parse speaker ID from filename: {filename}. Skipping.")
                    continue

        if not available_speaker_ids:
            audio_manager.print_and_speak(
                "No FastPitch speaker embeddings found. Please ensure they are generated."
            )
            print(
                f"No FastPitch speaker embeddings found in {config.FASTPITCH_EMBEDDINGS_DIR}.")
            return

        available_speaker_ids.sort()

        for i, speaker_id in enumerate(available_speaker_ids):
            print(f" {i}. FastPitch Speaker ID: {speaker_id}")

        print("\nType 'cancel' to return to the main menu.")

        selected_fastpitch_id = None
        while selected_fastpitch_id is None:
            choice_input = input(
                "Enter the number corresponding to the FastPitch Speaker ID you want to use: "
            ).strip().lower()

            if choice_input == 'cancel':
                audio_manager.print_and_speak("Voice selection cancelled.")
                return

            if choice_input.isdigit():
                index = int(choice_input)
                if 0 <= index < len(available_speaker_ids):
                    selected_fastpitch_id = available_speaker_ids[index]
                else:
                    audio_manager.print_and_speak(
                        "Invalid number. Please choose a number from the list.")
            else:
                audio_manager.print_and_speak(
                    "Invalid input. Please enter a number or 'cancel'.")

        audio_manager.print_and_speak(
            f"You have selected FastPitch Speaker ID {selected_fastpitch_id}."
        )
        print("\n--- Test the Selected Voice ---")

        audio_manager.print_and_speak(
            f"Now, let's hear FastPitch Speaker ID {selected_fastpitch_id}. Please type a sentence you'd like it to speak:"
        )
        test_text = input("Your text: ").strip()

        if test_text:
            audio_manager.print_and_speak(
                f"Speaking with FastPitch Speaker ID {selected_fastpitch_id}:"
            )
            audio_manager.synthesize_speech(
                text=test_text,
                speaker_id=selected_fastpitch_id,
                output_filename=f"test_fastpitch_speaker_{selected_fastpitch_id}.wav",
                output_dir=config.AUDIO_RECORDS_DIR
            )
            print(
                f"Test audio saved to {os.path.join(config.AUDIO_RECORDS_DIR, 'test_fastpitch_speaker_' + str(selected_fastpitch_id) + '.wav')}"
            )
        else:
            audio_manager.print_and_speak(
                "No text provided for testing. Skipping voice demonstration."
            )

    async def _synthesize_speech_from_text(self):
        """
        Handles the 'Synthesize speech from text' menu option, allowing choice
        between a default voice or a specific registered custom voice.
        """
        audio_manager.print_and_speak(
            "\nYou've chosen to synthesize speech from text.")

        text_to_synthesize = input("Enter text to synthesize: ").strip()
        if not text_to_synthesize:
            audio_manager.print_and_speak(
                "Text can't be empty. Returning to menu.")
            return

        # Initialize defaults
        selected_speaker_id = 0
        selected_speaker_embedding_np = None

        use_custom_voice_input = input(
            "Want to use a specific registered custom voice for this synthesis (overriding default)? (y/n): ").strip().lower()

        if use_custom_voice_input == 'y':
            print("\n--- Available Custom TTS Voices (from registered embeddings) ---")

            registered_tts_users = []
            # Gather user IDs from your embedding directory
            if os.path.isdir(config.EMBEDDING_OUTPUT_DIR):
                for filename in os.listdir(config.EMBEDDING_OUTPUT_DIR):
                    if filename.endswith("_embedding.pkl") or filename.endswith("_embedding.npy"):
                        user_id = filename.replace(
                            "_embedding.pkl", "").replace("_embedding.npy", "")
                        registered_tts_users.append(user_id)
            registered_tts_users.sort()

            if not registered_tts_users:
                audio_manager.print_and_speak(
                    "No custom TTS voices are registered. Can't use a specific personalization. Using the default voice instead.")
                print("No custom TTS voices found.")
                # We'll continue with the default selected_speaker_id=0 and None for embedding
            else:
                for i, user_id in enumerate(registered_tts_users):
                    print(f" {i+1}. {user_id}")

                chosen_user_id = None
                # Loop until a valid choice or 'cancel'
                while chosen_user_id is None:
                    temp_choice_input = input(
                        "Enter the number or ID of the speaker for this synthesis (or type 'cancel' to use default): ").strip().lower()

                    if temp_choice_input == 'cancel':
                        audio_manager.print_and_speak(
                            "Custom voice selection cancelled. Using the default voice.")
                        break  # Exit the loop, selected_speaker_embedding_np remains None

                    if temp_choice_input.isdigit():
                        index = int(temp_choice_input) - 1
                        if 0 <= index < len(registered_tts_users):
                            chosen_user_id = registered_tts_users[index]
                        else:
                            audio_manager.print_and_speak(
                                "That's an invalid number. Please choose a number from the list.")
                    else:
                        # Check if input matches any ID directly (case-insensitive)
                        for uid in registered_tts_users:
                            if uid.lower() == temp_choice_input:
                                chosen_user_id = uid
                                break
                        # If no match was found after checking all options
                        if chosen_user_id is None:
                            audio_manager.print_and_speak(
                                "I didn't understand that. Please enter a number, ID, or 'cancel'.")

                # If a valid user ID was chosen (not cancelled)
                if chosen_user_id:
                    # Attempt to load the selected speaker's embedding
                    filepath_pkl = os.path.join(
                        config.EMBEDDING_OUTPUT_DIR, f"{chosen_user_id}_embedding.pkl")
                    filepath_npy = os.path.join(
                        config.EMBEDDING_OUTPUT_DIR, f"{chosen_user_id}_embedding.npy")

                    embedding_to_load = None
                    if os.path.exists(filepath_pkl):
                        try:
                            with open(filepath_pkl, 'rb') as f:
                                embedding_to_load = pkl.load(f)
                            if isinstance(embedding_to_load, torch.Tensor):
                                embedding_to_load = embedding_to_load.cpu().numpy()
                            selected_speaker_embedding_np = embedding_to_load.flatten()
                        except Exception as e:
                            print(
                                f"Error loading pickle embedding for {chosen_user_id}: {e}")
                    elif os.path.exists(filepath_npy):
                        try:
                            embedding_to_load = np.load(filepath_npy)
                            selected_speaker_embedding_np = embedding_to_load.flatten()
                        except Exception as e:
                            print(
                                f"Error loading numpy embedding for {chosen_user_id}: {e}")

                    if selected_speaker_embedding_np is not None:
                        user_id_for_synthesis_display = chosen_user_id
                        audio_manager.print_and_speak(
                            f"Okay, I'll use the custom voice for '{user_id_for_synthesis_display}' for this synthesis.")
                    else:
                        audio_manager.print_and_speak(
                            "Couldn't load the embedding for your selection. I'll use the default voice instead.")
        else:
            audio_manager.print_and_speak(
                "Using the default voice for synthesis.")

        # Determine output filename
        output_filename_tts = input(
            "Enter the desired output audio filename (e.g., 'my_tts_output.wav'): ").strip()
        if not output_filename_tts:
            output_filename_tts = "synthesized_output.wav"
        if not output_filename_tts.endswith('.wav'):
            output_filename_tts += ".wav"

        # Call synthesize_speech: it will use either the custom embedding (if not None)
        # or the default speaker_id (which is 0 by default)
        audio_manager.synthesize_speech(
            text=text_to_synthesize,
            # This will be 0 if no custom embedding is chosen/loaded
            speaker_id=selected_speaker_id,
            # This will be None if no custom embedding is chosen/loaded
            speaker_embedding_np=selected_speaker_embedding_np,
            output_filename=output_filename_tts,
            output_dir=audio_manager.audio_records_dir
        )
        audio_manager.print_and_speak(f"Speech synthesized and saved.")
        print(
            f"Speech synthesized to {os.path.join(audio_manager.audio_records_dir, output_filename_tts)}")

    def _list_registered_speakers(self):
        """Handles the 'List all registered speaker embeddings' menu option."""
        audio_manager.print_and_speak(
            "\nYou have chosen to list registered speaker embeddings")

        # List speaker verification embeddings (from config.EMBEDDING_OUTPUT_DIR)
        print("\n--- Speaker Verification Embeddings (for authentication/custom TTS) ---")

        # Local implementation of getting registered TTS user IDs
        verification_user_ids = []
        if os.path.isdir(config.EMBEDDING_OUTPUT_DIR):
            for filename in os.listdir(config.EMBEDDING_OUTPUT_DIR):
                if filename.endswith("_embedding.pkl") or filename.endswith("_embedding.npy"):
                    user_id = filename.replace(
                        "_embedding.pkl", "").replace("_embedding.npy", "")
                    verification_user_ids.append(user_id)
        verification_user_ids.sort()  # Sort for presentation

        if verification_user_ids:
            for i, user_id in enumerate(verification_user_ids):
                print(f" {i+1}. {user_id}")
            speakable_list_verification = ", ".join(verification_user_ids)
            audio_manager.print_and_speak(
                f"Currently registered speakers for verification and custom TTS: {speakable_list_verification}.")
        else:
            print("No custom speaker embeddings registered yet.")
            audio_manager.print_and_speak(
                "No custom speaker embeddings registered yet for verification or custom TTS.")

    def _delete_speaker_embedding(self):
        """
        Handles the 'Delete a registered speaker embedding' menu option.
        First lists available embeddings, then prompts the user to select one by number for deletion.
        """
        audio_manager.print_and_speak(
            "\nYou have chosen to delete a speaker embedding.")

        # Get the list of registered speaker IDs
        registered_speakers = []
        if os.path.isdir(config.EMBEDDING_OUTPUT_DIR):
            for filename in os.listdir(config.EMBEDDING_OUTPUT_DIR):
                if filename.endswith("_embedding.pkl") or filename.endswith("_embedding.npy"):
                    user_id = filename.replace(
                        "_embedding.pkl", "").replace("_embedding.npy", "")
                    registered_speakers.append(user_id)
        registered_speakers.sort()

        if not registered_speakers:
            audio_manager.print_and_speak(
                "No speaker embeddings registered yet to delete. Returning to menu.")
            print("No speaker embeddings found for deletion.")
            return

        # Display the numbered list of speakers
        print("\n--- Registered Speaker Embeddings for Deletion ---")
        for i, user_id in enumerate(registered_speakers):
            print(f" {i+1}. {user_id}")
        print("\nType 'back' to return to the main menu.")

        user_id_to_delete = None
        while user_id_to_delete is None:
            choice_input = input(
                "Enter the number corresponding to the speaker to delete, or type 'back' to return to the menu: "
            ).strip().lower()

            if choice_input == 'back':
                audio_manager.print_and_speak(
                    "Deletion cancelled. Returning to menu.")
                return

            if choice_input.isdigit():
                index = int(choice_input) - 1
                if 0 <= index < len(registered_speakers):
                    user_id_to_delete = registered_speakers[index]
                else:
                    audio_manager.print_and_speak(
                        "Invalid number. Please choose a number from the list.")
            else:
                audio_manager.print_and_speak(
                    "Invalid input. Please enter a number or 'back'.")

        # Proceed with deletion for the selected user_id
        embedding_filepath_pkl = os.path.join(
            config.EMBEDDING_OUTPUT_DIR, f"{user_id_to_delete}_embedding.pkl")
        embedding_filepath_npy = os.path.join(
            config.EMBEDDING_OUTPUT_DIR, f"{user_id_to_delete}_embedding.npy")

        found_file = False
        if os.path.exists(embedding_filepath_pkl):
            os.remove(embedding_filepath_pkl)
            found_file = True
            print(
                f"Successfully deleted embedding file for '{user_id_to_delete}' (.pkl).")

        if os.path.exists(embedding_filepath_npy):
            os.remove(embedding_filepath_npy)
            found_file = True
            print(
                f"Successfully deleted embedding file for '{user_id_to_delete}' (.npy).")

        if found_file:
            audio_manager.print_and_speak(
                f"Speaker '{user_id_to_delete}' embedding deleted successfully.")
        else:
            # This case should ideally not be hit if user_id_to_delete was successfully picked from the list
            # but it's a fallback for robustness.
            audio_manager.print_and_speak(
                f"No embedding files found for '{user_id_to_delete}'.")

        audio_manager.print_and_speak(
            f"Deletion process for '{user_id_to_delete}' complete.")

    def _clear_all_speaker_embeddings(self):
        """Handles the 'Clear ALL registered speaker embeddings' menu option."""
        audio_manager.print_and_speak(
            "\nYou have chosen to clear all speaker embeddings")
        confirm = input(
            "Are you sure you want to delete ALL registered speaker embeddings? This cannot be undone! (yes/no): ").strip().lower()
        if confirm == 'yes':
            try:
                # Clear all files from the embedding output directory
                if os.path.exists(config.EMBEDDING_OUTPUT_DIR):
                    shutil.rmtree(config.EMBEDDING_OUTPUT_DIR)
                # Recreate the directory
                os.makedirs(config.EMBEDDING_OUTPUT_DIR, exist_ok=True)
                print(
                    f"All speaker embeddings cleared from {config.EMBEDDING_OUTPUT_DIR}.")

                audio_manager.print_and_speak(
                    "All registered speaker embeddings have been cleared.")
            except Exception as e:
                audio_manager.print_and_speak(
                    f"Error clearing embeddings: {e}")
                print(f"Error clearing embeddings: {e}")
        else:
            audio_manager.print_and_speak(
                "Operation cancelled. No embeddings were deleted.")

    def _toggle_tts_for_answers(self):
        """Handles the 'Toggle TTS for computer answers' menu option."""
        # This directly modifies the audio_manager's internal flag
        audio_manager.use_tts_for_answers_flag = not audio_manager.use_tts_for_answers_flag
        status = "ENABLED" if audio_manager.use_tts_for_answers_flag else "DISABLED"
        audio_manager.print_and_speak(
            f"Text to Speech for computer answers is now {status}.")
