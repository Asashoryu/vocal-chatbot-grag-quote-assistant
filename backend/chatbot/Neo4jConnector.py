# NEW: Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError


class Neo4jConnector:
    """
    Manages the connection to the Neo4j database.
    Encapsulates driver initialization and closing, ensuring a single driver instance
    is maintained and its connectivity is verified before use.
    """

    def __init__(self, uri, username, password):
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = None  # Initialize driver as None

    def connect(self):
        """
        Establishes a connection to Neo4j.
        If a driver already exists, it verifies connectivity. If connectivity is lost,
        it attempts to re-establish the connection.

        Returns:
            bool: True if connection is successful or already established and verified,
                  False otherwise.
        """
        if self._driver is None:
            # No driver exists, attempt to create one
            try:
                self._driver = GraphDatabase.driver(
                    self._uri, auth=(self._username, self._password))
                # Verify connectivity right after creation
                self._driver.verify_connectivity()
                print("Successfully connected to Neo4j.")
                return True
            except ServiceUnavailable as e:
                print(
                    f"Connection failed: Neo4j database is not running or accessible at {self._uri}. Error: {e}")
                self._driver = None  # Ensure driver is reset on failure
                return False
            except Exception as e:
                print(f"Failed to establish initial connection to Neo4j: {e}")
                print("Please check your URI, username, and password.")
                self._driver = None  # Ensure driver is reset on failure
                return False
        else:
            # Driver already exists, verify its connectivity
            try:
                self._driver.verify_connectivity()
                # print("Neo4j driver is already connected and verified.") # Optional: for verbose logging
                return True
            except (ServiceUnavailable, ClientError) as e:
                # Connection lost or became unavailable
                print(
                    f"Lost connection to Neo4j: {e}. Attempting to re-establish connection...")
                self._driver.close()  # Close the stale driver
                self._driver = None  # Reset to None to force a new connection attempt
                return self.connect()  # Recursively call connect to try again
            except Exception as e:
                print(
                    f"An unexpected error occurred while verifying Neo4j connection: {e}")
                self._driver.close()  # Close the potentially problematic driver
                self._driver = None  # Reset
                return False

    def close(self):
        """
        Closes the Neo4j driver if it exists and is open.
        """
        if self._driver:
            try:
                self._driver.close()
                print("Neo4j driver closed.")
            except Exception as e:
                print(f"Error closing Neo4j driver: {e}")
            finally:
                self._driver = None  # Ensure driver reference is cleared

    def get_driver(self):
        """
        Returns the initialized Neo4j driver instance.
        It does NOT attempt to connect if not already connected.
        It's assumed that `connect()` has been called successfully prior to this.
        """
        return self._driver
