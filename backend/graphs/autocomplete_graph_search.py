from neo4j import GraphDatabase
# Corrected import for ServiceUnavailable
from neo4j.exceptions import ServiceUnavailable

# --- Configuration ---
# IMPORTANT: Replace these with your actual Neo4j connection details.
# If running locally with default settings, these might be correct.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
# <--- REPLACE WITH YOUR ACTUAL NEO4J PASSWORD
NEO4J_PASSWORD = "amonolexandr"

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


def main():
    """
    Main function to handle user interaction and orchestrate the search process.
    """
    connector = Neo4jConnector(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    if not connector.connect():
        return  # Exit if connection fails

    try:
        while True:
            print("\n--- Select Entity Type for Search ---")
            for key, info in ENTITY_INDEX_MAP.items():
                print(f"  {key}: {info['name']}")
            print("  (Type 'exit' to quit)")

            choice = input(
                "Enter your choice (1-5 or 'exit'): ").strip().lower()

            if choice == 'exit':
                break

            if choice not in ENTITY_INDEX_MAP:
                print("Invalid choice. Please enter a number from 1 to 5.")
                continue

            substring = input(
                f"Enter substring to search for in {ENTITY_INDEX_MAP[choice]['name']}: ").strip()
            if not substring:
                print("Substring cannot be empty. Please try again.")
                continue

            try:
                k = int(input("Enter number of top results (default 10): ") or "10")
                if k <= 0:
                    print("Number of results must be positive. Defaulting to 10.")
                    k = 10
            except ValueError:
                print("Invalid number. Defaulting to 10 results.")
                k = 10

            search_results = search_entity(
                connector.get_driver(), choice, substring, k)

            if search_results:
                print(
                    f"\n--- Search Results for '{substring}' in {ENTITY_INDEX_MAP[choice]['name']} ---")
                for i, res in enumerate(search_results):
                    print(
                        f"{i+1}. Type: {res['NodeType']}, Content: {res['Content']}, Score: {res['Score']:.2f}")
            else:
                print("No results found for your search criteria.")

    finally:
        connector.close()


if __name__ == "__main__":
    main()
