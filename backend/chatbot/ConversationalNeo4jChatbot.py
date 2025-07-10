# conversational_neo4j_chatbot.py

import os
import soundfile as sf
import pickle as pkl
import asyncio
import numpy as np
from typing import Optional

# --- Import the globally accessible AudioManager instance ---
from AudioManager import audio_manager

from Neo4jConnector import Neo4jConnector
import ollama


class ConversationalNeo4jChatbot:
    """
    Manages LLM conversational interactions, querying the Neo4j graph,
    and utilizing the AudioManager for vocal interactions.
    This class should be instantiated once per conversation session.
    """

    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str,
                 entity_index_map: dict, llm_call_function=None):
        """
        Initializes the conversational chatbot.
        It directly uses the AudioManager singleton for audio operations.

        Args:
            neo4j_uri (str): URI for the Neo4j database.
            neo4j_username (str): Username for Neo4j authentication.
            neo4j_password (str): Password for Neo4j authentication.
            entity_index_map (dict): Mapping of entity types to Neo4j full-text indexes.
            llm_call_function (callable, optional): The function to call the local LLM.
                                                    If None, the internal _default_ollama_call will be used.
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.entity_index_map = entity_index_map

        # Stores the context from the first query
        self._current_graph_context = "No specific content found yet from the initial query."

        # Initialize Ollama client for the default LLM call
        try:
            self._ollama_client = ollama.AsyncClient(
                host='http://localhost:11434')
            # Ensure this model is pulled "deepseek-r1:1.5b" "tinyllama" "gemma:2b"
            self.ollama_model_name = "deepseek-r1:1.5b"
            print(
                f"Ollama client initialized with default model '{self.ollama_model_name}'.")
        except Exception as e:
            print(
                f"Error initializing Ollama client: {e}. LLM calls might fail if default is used.")
            self._ollama_client = None
            self.ollama_model_name = None

        if llm_call_function is None:
            self.llm_call_function = self._default_ollama_call
            print("Using internal _default_ollama_call for LLM interactions.")
        else:
            self.llm_call_function = llm_call_function
            print("Using externally provided llm_call_function for LLM interactions.")

        self._conversation_history = []

        # New instance variables for current session's user and embedding
        self._current_user_name: str = "Guest"  # Initialize with a default value
        self._current_speaker_embedding: Optional[np.ndarray] = None

        self.neo4j_connector = Neo4jConnector(
            self.neo4j_uri, self.neo4j_username, self.neo4j_password)

        print(
            f"DEBUG: Chatbot initialized with entity_index_map: {self.entity_index_map}")

        self.spoken_exit_commands = {
            "two": "two",
            "exit": "two",
            "one": "one",
            "": ""  # Handle empty transcription
        }

    async def _default_ollama_call(self, prompt: str) -> str:
        """
        Interacts with a local Ollama LLM to generate a response based on the prompt.
        This is the internal, default LLM function for the chatbot if no other is provided.
        It specifically extracts and returns only the final 'content' from DeepSeek's
        potentially structured output, discarding any 'thinking' parts.
        """
        if not self._ollama_client:
            print(
                "DEBUG: _default_ollama_call: Ollama client not initialized. Cannot make LLM call.")
            return "Error: Ollama client not initialized. Cannot make LLM call."
        if not self.ollama_model_name:
            print(
                "DEBUG: _default_ollama_call: Ollama model name not set. Cannot make LLM call.")
            return "Error: Ollama model name not set. Cannot make LLM call."

        try:
            response = await self._ollama_client.chat(
                model=self.ollama_model_name,
                messages=[
                    {'role': 'user', 'content': prompt},
                ],
                stream=False  # Ensures a single, complete response object
            )

            llm_response_content = ""  # Use a new variable name to avoid confusion

            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                llm_response_content = response.message.content

            if llm_response_content:
                return llm_response_content
            else:
                print(
                    "DEBUG: _default_ollama_call: Ollama response content was empty or structure unexpected.")
                return "Error: Ollama response structure was unexpected or content was empty."

        except ollama.ResponseError as e:
            print(
                f"ERROR: _default_ollama_call: Error interacting with Ollama: {e}.")
            return f"Error interacting with Ollama: {e}. Please ensure the Ollama server is running and the model '{self.ollama_model_name}' is pulled locally."
        except Exception as e:
            print(
                f"ERROR: _default_ollama_call: An unexpected error occurred while calling the local LLM: {e}. Check Ollama server status or network.")
            return f"An unexpected error occurred while calling the local LLM: {e}. Check Ollama server status or network."

    async def _process_vocal_input(self, prompt_message: str, user_id: str) -> str:
        """
        Handles recording and transcribing user's vocal input using AudioManager.
        Returns the transcribed text or an empty string if transcription fails.
        Now takes user_id for better temporary file naming.
        """
        recorded_audio_data = audio_manager.record_audio(
            prompt_message=prompt_message
        )

        transcribed_text = ""
        temp_audio_filepath = None

        if recorded_audio_data is not None:
            try:
                # Use user_id in temp filename to prevent conflicts if multiple users are handled concurrently
                temp_output_filename = "temp_vocal_input.wav"
                temp_audio_filepath = os.path.join(
                    audio_manager.audio_records_dir, temp_output_filename)

                os.makedirs(audio_manager.audio_records_dir,
                            exist_ok=True)
                sf.write(temp_audio_filepath, recorded_audio_data,
                         audio_manager.sample_rate)
                print(
                    f"DEBUG: _process_vocal_input: Temporary audio saved: {temp_audio_filepath}")

                transcribed_text = audio_manager.transcribe_audio(
                    temp_audio_filepath)

                # >>> DEBUG: Print the transcribed text
                print(
                    f"DEBUG: _process_vocal_input: Transcribed text: '{transcribed_text}'")

                if not transcribed_text:
                    audio_manager.print_and_speak(
                        "I couldn't transcribe your speech. Please try again.")  # Use current embedding
            finally:
                if temp_audio_filepath and os.path.exists(temp_audio_filepath):
                    os.remove(temp_audio_filepath)
                    print(
                        f"DEBUG: _process_vocal_input: Cleaned up temporary audio: {temp_audio_filepath}")
        else:
            audio_manager.print_and_speak(
                "No audio recorded. Please try again.")  # Use current embedding
            print("DEBUG: _process_vocal_input: No audio data recorded.")

        return transcribed_text.strip()

    def _search_entity(self, driver, entity_choice: str, substring: str, k: int = 10):
        """
        Performs a full-text search specifically for 'Quote' nodes and
        then retrieves their connected Author, Context, and Detail nodes.
        """
        print("DEBUG: Entering _search_entity method.")
        if not driver:
            print(
                "ERROR: _search_entity: Neo4j driver is not initialized. Returning empty list.")
            return []

        quote_index_info = next(
            (v for k_map, v in self.entity_index_map.items() if v.get("name") == "Quote"), None)

        if not quote_index_info:
            print("ERROR: _search_entity: 'Quote' index information not found in entity_index_map. "
                  "Please ensure it contains an entry like {'name': 'Quote', 'index': 'quoteTextsIndex'}.")
            return []

        index_name = quote_index_info["index"]
        entity_name = quote_index_info["name"]

        print(
            f"DEBUG: _search_entity: Using index: '{index_name}', entity: '{entity_name}', raw substring: '{substring}'")

        results = []

        with driver.session() as session:
            if not substring or not substring.strip():
                print(
                    "DEBUG: _search_entity: Substring is empty or whitespace. Returning empty results.")
                return []

            search_term = f"{substring}*"

            query = f"""
            CALL db.index.fulltext.queryNodes('{index_name}', $searchTerm) YIELD node AS quote, score
            WHERE 'Quote' IN labels(quote)
            OPTIONAL MATCH (author:Author)-[:WROTE]->(quote)
            OPTIONAL MATCH (context:Context)<-[:HAS_CONTEXT]-(quote)
            OPTIONAL MATCH (detail:Detail)<-[:HAS_DETAIL]-(quote)
            RETURN
                quote.text AS QuoteContent,
                score,
                COLLECT(DISTINCT author.name) AS Authors,
                COLLECT(DISTINCT context.text) AS Contexts,
                COLLECT(DISTINCT detail.text) AS Details
            ORDER BY score DESC
            LIMIT $k
            """
            print(f"\nDEBUG: _search_entity: Executing Cypher Query:\n{query}")
            print(
                f"DEBUG: _search_entity: With parameters: searchTerm='{search_term}', k={k}")

            try:
                result = session.run(query, searchTerm=search_term, k=k)
                found_records_count = 0
                for record in result:
                    found_records_count += 1

                    authors = [a for a in record["Authors"] if a is not None]
                    contexts = [c for c in record["Contexts"] if c is not None]
                    details = [d for d in record["Details"] if d is not None]

                    processed_record = {
                        "QuoteContent": record["QuoteContent"],
                        "Score": record["score"],
                        "Authors": authors,
                        "Contexts": contexts,
                        "Details": details
                    }
                    results.append(processed_record)

                if found_records_count == 0:
                    print(
                        f"DEBUG: _search_entity: Neo4j query returned no records for substring '{substring}'.")

            except Exception as e:
                print(
                    f"ERROR: _search_entity: Caught an exception during Neo4j query execution: {e}")
                print("INFO: Please ensure the full-text index is correctly named and your graph schema matches relationships and properties.")
                return []  # Return empty list on error

        print(f"DEBUG: _search_entity: Returning {len(results)} results.")
        return results

    async def _extract_llm_parts(self, llm_raw_output: str) -> tuple[str, str]:
        """
        Extracts the 'thinking' and 'answer' parts from the LLM's raw output
        based on <think> tags.
        """
        thinking_start_tag = "<think>"
        thinking_end_tag = "</think>"

        thinking_start_index = llm_raw_output.find(thinking_start_tag)
        thinking_end_index = llm_raw_output.find(thinking_end_tag)

        thinking_part = ""
        answer_part = llm_raw_output

        if thinking_start_index != -1 and thinking_end_index != -1 and thinking_end_index > thinking_start_index:
            thinking_part = llm_raw_output[thinking_start_index +
                                           len(thinking_start_tag): thinking_end_index].strip()
            answer_part = llm_raw_output[thinking_end_index +
                                         len(thinking_end_tag):].strip()

        print(
            f"DEBUG: _extract_llm_parts: Thinking part: \n'{thinking_part}\n...'")
        print(
            f"DEBUG: _extract_llm_parts: Answer part: \n'{answer_part}\n...'")

        return thinking_part, answer_part

    async def _generate_llm_response(self, user_query: str, force_graph_query: bool = True, user_id: str = "Guest") -> str:
        """
        Queries Neo4j for context (if force_graph_query is True) or uses existing context,
        and generates an LLM response, incorporating conversation history.
        Now takes user_id for potential contextualization in the LLM.
        """
        print(
            f"DEBUG: _generate_llm_response: Processing user query for '{user_id}': '{user_query}' (Force Graph Query: {force_graph_query})")

        # Ensure connection before proceeding with the query (still needed even if not querying, for initial setup checks)
        if not self.neo4j_connector.connect():
            error_message = "I'm sorry, I cannot connect to the knowledge base at this moment. Please try again later."
            print(
                f"ERROR: _generate_llm_response: Could not connect to Neo4j. {error_message}")
            return error_message

        context_string_for_llm = ""

        if force_graph_query:
            try:
                search_term = user_query
                print(
                    f"DEBUG: _generate_llm_response: Initial user_query: '{user_query}'")
                audio_manager.print_and_speak(f"User asked: '{user_query}'")
                if ("quote:" in user_query.lower() or "said:" in user_query.lower()) and len(user_query) > 10:
                    search_term = user_query.split(':', 1)[-1].strip()
                print(
                    f"DEBUG: _generate_llm_response: Derived search_term for Neo4j: '{search_term}'")

                if not search_term.strip():
                    print(
                        "DEBUG: _generate_llm_response: Search term is empty after processing. Skipping Neo4j search.")
                    context_string_for_llm = "No specific content found in the graph relevant to the query as the search term was empty."
                else:
                    search_results = self._search_entity(
                        self.neo4j_connector.get_driver(), None, search_term, k=5)

                    if search_results:
                        formatted_results = []
                        for i, res in enumerate(search_results):
                            quote_content = res.get('QuoteContent') or "N/A"
                            score = res.get('Score', 0.0)
                            quote_info = f"Quote {i+1} (Score: {score:.2f}): \"{quote_content}\""

                            if res.get('Authors'):
                                quote_info += f"\n  Authors: {', '.join(res['Authors'])}"
                            if res.get('Contexts'):
                                quote_info += f"\n  Contexts: {'; '.join(res['Contexts'])}"
                            if res.get('Details'):
                                quote_info += f"\n  Details: {'; '.join(res['Details'])}"

                            formatted_results.append(quote_info)
                        context_string_for_llm = "\n\n".join(formatted_results)
                    else:
                        context_string_for_llm = "No specific content found in the graph relevant to the query."
                        print(
                            f"DEBUG: _generate_llm_response: No search results found. Context set to: '{context_string_for_llm}'")

            except Exception as e:
                error_message = f"An error during Neo4j retrieval happened; no answer will be generated for the query."
                print(
                    f"ERROR: _generate_llm_response: Exception caught during Neo4j retrieval: {e}")
                context_string_for_llm = "An error occurred while searching the knowledge base. Please try rephrasing your question."
            finally:
                pass

            # Store the context for subsequent queries
            self._current_graph_context = context_string_for_llm
            print(
                f"DEBUG: _generate_llm_response: Stored _current_graph_context for follow-ups.")
        else:
            # Use the previously stored context for follow-up questions
            context_string_for_llm = self._current_graph_context
            print(
                f"DEBUG: _generate_llm_response: Using stored _current_graph_context for follow-up.")

        chat_history_str = "\n".join(
            self._conversation_history) if self._conversation_history else "No previous conversation."

        llm_prompt = f"""You are a human that quotes the best matching result and summarizes it's meaning from a graph database of quotes.
        You should also take into account the previous conversation to answer follow-up questions. IMPORTANT: BE VERY CONCISE OR I WILL KILL YOU!

        --- Previous Conversation ---
        {chat_history_str}
        -----------------------------

        My query was: "{user_query}"
        The results from the Neo4j Graph where:

        "{context_string_for_llm}"

        ; tldr;
        """
        print(
            f"DEBUG: _generate_llm_response: Final LLM prompt:\n\n{llm_prompt}...")

        llm_raw_response = await self.llm_call_function(llm_prompt)

        thinking_process, final_answer = await self._extract_llm_parts(llm_raw_response)

        print(
            f"DEBUG: _generate_llm_response: Returning final answer: \n\n'{final_answer}'")
        return final_answer

    async def start_conversation_loop(self, initial_user_name: Optional[str] = None):
        """
        Starts the main conversational loop for Neo4j queries.
        It relies on the global speaker ID set in AudioManager for TTS responses.
        """
        # Set the current user for this conversation session
        self._current_user_name = initial_user_name if initial_user_name else "Guest"

        audio_manager.print_and_speak(
            f"Hello {self._current_user_name}. What's your initial question for the knowledge graph?"
        )
        print(
            f"You can say 'two' at any time to go back to the main menu, {self._current_user_name}.")

        while True:
            # Clear history for a new initial query session
            self._conversation_history = []
            # Clear the stored graph context for a fresh start (Reset context)
            self._current_graph_context = "No specific content found yet from the initial query."

            audio_manager.print_and_speak(
                "Ask your initial question to the knowledge graph.")

            initial_query_text = await self._process_vocal_input(
                prompt_message=f"Ask your initial question (or 'two') for {audio_manager.recording_duration_seconds} seconds.",
                user_id=self._current_user_name  # Pass the current user ID
            )

            lower_initial_query = initial_query_text.lower().strip()
            if self.spoken_exit_commands.get(lower_initial_query) == "two":
                audio_manager.print_and_speak(
                    "Returning to the main menu.")
                return
            elif not lower_initial_query:  # Check for empty string after strip()
                audio_manager.print_and_speak(
                    "I didn't hear a question. Please try again.")
                continue

            audio_manager.print_and_speak(
                "Processing your question...")
            # Call with force_graph_query=True for the initial query
            chatbot_response = await self._generate_llm_response(
                initial_query_text,
                force_graph_query=True,
                user_id=self._current_user_name  # Pass current user ID
            )
            audio_manager.print_and_speak(
                f"\n--- Chatbot: {chatbot_response}")

            # Add the initial turn to conversation history
            self._conversation_history.append(
                f"{self._current_user_name}: {initial_query_text}")
            self._conversation_history.append(f"Chatbot: {chatbot_response}")

            # Enter the continuous follow-up conversation loop
            while True:
                audio_manager.print_and_speak(
                    "Would you like to know more about it? Or say 'one' to ask a new question, or 'two' to go to the main menu."
                )

                follow_up_query_text = await self._process_vocal_input(
                    prompt_message=f"Speak your follow-up question (or 'one' or 'two') for {audio_manager.recording_duration_seconds} seconds.",
                    user_id=self._current_user_name  # Pass the current user ID
                )

                lower_follow_up_query: str = follow_up_query_text.lower().strip()
                if self.spoken_exit_commands.get(lower_follow_up_query) == "one":
                    audio_manager.print_and_speak(
                        "Okay, let's go back to asking a new question for the knowledge graph.")
                    # Exit inner (follow-up) loop, go back to outer loop for new initial query
                    break
                elif self.spoken_exit_commands.get(lower_follow_up_query) == "two":
                    audio_manager.print_and_speak(
                        "Returning to the main menu.")
                    return  # Exit both loops and return to interactive_main's top level
                elif not lower_follow_up_query:  # Check for empty string after strip()
                    audio_manager.print_and_speak(
                        "I didn't hear a question. Please try again.")
                    continue

                # Process the spoken follow-up query
                audio_manager.print_and_speak(
                    "Processing your follow-up question...")
                # Call with force_graph_query=False for follow-up questions
                chatbot_response = await self._generate_llm_response(
                    follow_up_query_text,
                    force_graph_query=False,
                    user_id=self._current_user_name  # Pass current user ID
                )
                audio_manager.print_and_speak(
                    f"\n--- Chatbot: {chatbot_response}")

                # Add the follow-up turn to conversation history
                self._conversation_history.append(
                    f"User: {follow_up_query_text}")
                self._conversation_history.append(
                    f"Chatbot: {chatbot_response}")
