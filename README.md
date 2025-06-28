# Wikiquote Graph-Based Vocal Chatbot

A complete system that builds a knowledge graph from Wikiquote, enables smart quote autocompletion, and interacts with users through voice using speaker identification and personalized TTS.

---

## Project Scope

This project is part of the NLP final assignment and involves **two major components**:

### Step 1 – Wikiquote Graph RAG Autocomplete

- Extract quotes and metadata from the [Wikiquote XML dump (June 2025)](https://dumps.wikimedia.org/enwikiquote/20250601)
- Build a knowledge graph using Neo4j
- Create a full-text and/or vector index over the quotes
- Implement an **autocomplete system** that returns best-matching quotes with their source

### Step 2 – "Which Quote?" Vocal Chatbot

- Users speak to the system
- Automatic Speech Recognition (ASR) transcribes the input
- Speaker Identification detects the user
- Personalized **Text-to-Speech (TTS)** replies with quote results using a voice tailored to the recognized speaker
