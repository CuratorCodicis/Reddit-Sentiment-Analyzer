import pymongo  # MongoDB driver to connect and interact with the database
from config import MONGODB

# MongoDB connection details
MONGO_URI = MONGODB["uri"]
MONGO_DB_NAME = MONGODB["db_name"]


def get_mongo_client():
    """
    Initialize and return a MongoDB client using the MONGO_URI.

    This function tries to connect to MongoDB and prints a success message,
    or an error if the connection fails.
    """
    try:
        # Initialize the MongoDB client
        client = pymongo.MongoClient(MONGO_URI)
        print(f"✅ Successfully connected to MongoDB: {MONGO_DB_NAME}")
        return client
    except Exception as e:
        # If there is a connection error, print an error message
        print(f"❌ MongoDB Connection Error: {e}")
        return None


def get_database(client):
    """
    Given a MongoDB client, return the database specified by MONGO_DB_NAME.

    :param client: The pymongo.MongoClient instance.
    :return: The database object or None if the client is not available.
    """
    if client:
        return client[MONGO_DB_NAME]
    else:
        return None


def list_collections(db):
    """
    List all collections in the given database and print them.

    :param db: The MongoDB database object.
    :return: A list of collection names.
    """
    try:
        collections = db.list_collection_names()
        print(f"📂 Existing collections: {collections}")
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []


def insert_documents(collection_name, documents):
    """
    Inserts multiple documents into the specified collection.
    If the collection does not exist, MongoDB will create it automatically.

    :param collection_name: Name of the collection (e.g., "posts")
    :param documents: List of dictionaries to be inserted.
    :return: List of inserted document IDs (in the database) or None if an error occurs.
    """
    client = get_mongo_client()
    if client is None:
        print("❌ Cannot insert documents: MongoDB client is unavailable.")
        return None
    db = get_database(client)
    collection = db[collection_name]
    try:
        # insert_many returns an object with inserted_ids attribute
        result = collection.insert_many(documents, ordered=False)
        print(
            f"✅ Inserted {len(result.inserted_ids)} documents into '{collection_name}' collection"
        )
        return result.inserted_ids
    except Exception as e:
        print(f"❌ Error inserting documents: {e}")
    finally:
        client.close()  # Closing the DB connection


def fetch_documents(collection_name, query={}, projection=None):
    """
    Fetches documents from the specified collection based on a query.

    :param collection_name: Name of the collection (e.g., "posts")
    :param query: A dictionary specifying MongoDB query filters (default: {} retrieves all documents, {"id": "abc123"} for a specific reddit-id).
    :param projection: A dictionary specifying fields to include or exclude (e.g. {"title": 1, "author": 1} to return only certain fields.).
    :return: List of documents (each as a dictionary) or None if an error occurs.
    """
    client = get_mongo_client()
    if client is None:
        print("❌ Cannot fetch documents: MongoDB client is unavailable.")
        return None
    db = get_database(client)
    collection = db[collection_name]  # i.e. 'posts', 'comments', ...
    try:
        # Convert the cursor to a list to fetch all matching documents
        documents = list(collection.find(query, projection))
        print(
            f"✅ Fetched {len(documents)} documents from '{collection_name}' collection."
        )
        return documents
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
        return None
    finally:
        client.close()  # Closing the DB connection


def create_unique_indexes():
    """
    Create unique indexes on the 'id' field for both the 'posts' and 'comments' collections.
    This prevents duplicate entries when inserting documents.
    """
    client = get_mongo_client()
    if client:
        db = get_database(client)
        try:
            # Create a unique index on the "id" field in the "posts" colelction
            posts_index = db["posts"].create_index("id", unique=True)
            print(f"✅ Unique index created for 'posts' collection: {posts_index}")

            # Create a unique index on the "id" field in the "comments" collection
            comments_index = db["comments"].create_index("id", unique=True)
            print(
                f"✅ Unique index created for 'comments' collection: {comments_index}"
            )
        except Exception as e:
            print(f"❌ Error creating indexes: {e}")
        finally:
            client.close()


# Test code: This block runs when you execute database.py directly.
if __name__ == "__main__":
    # Create a client and get the database
    client = get_mongo_client()
    if client:
        db = get_database(client)
        # List current collections for debugging purposes
        list_collections(db)
        # Example: Insert a sample post document into a "posts" collection
        sample_posts = [
            {
                "id": "abc123",
                "title": "Sample Reddit Post",
                "selftext": "This is a sample post for testing.",
                "author": "sample_user",
                "score": 42,
                "num_comments": 5,
                "created_utc": "2023-01-01T12:00:00Z",
                "url": "http://example.com",
            }
        ]
        insert_documents("posts", sample_posts)
        # Fetch documents back to verify insertion
        fetched = fetch_documents("posts")
        print("Fetched Documents:")
        for doc in fetched:
            print(doc)
