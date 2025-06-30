import json
from neo4j import GraphDatabase

# DON'T USE THIS SCRIPT: TOO SLOW !!

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "amonolexandr"

# Path to your JSONL file
JSONL_PATH = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/json/quotes_output.jsonl"


class QuoteGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_graph(self, jsonl_path):
        with self.driver.session() as session:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        session.execute_write(self._add_quote, data)

    @staticmethod
    def _add_quote(tx, data):
        author = data.get("author")
        quote = data.get("quote")
        context = data.get("context")
        sources = data.get("sources", [])

        # Merge Author and Quote and link them
        tx.run("""
            MERGE (a:Author {name: $author})
            MERGE (q:Quote {text: $quote})
            MERGE (a)-[:WROTE]->(q)
        """, author=author, quote=quote)

        # Merge Context and link it to Quote
        if context:
            tx.run("""
                MERGE (c:Context {text: $context})
                WITH c
                MATCH (q:Quote {text: $quote})
                MERGE (q)-[:HAS_CONTEXT]->(c)
            """, quote=quote, context=context)

        # Merge Sources and link them to Quote
        for source in sources:
            source_text = source.get("text")
            if source_text:
                tx.run("""
                    MERGE (s:Detail {text: $source_text})
                    WITH s
                    MATCH (q:Quote {text: $quote})
                    MERGE (q)-[:HAS_DETAIL]->(s)
                """, quote=quote, source_text=source_text)


if __name__ == "__main__":
    builder = QuoteGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    builder.create_graph(JSONL_PATH)
    builder.close()
    print("✅ Graph successfully built from JSONL file.")
